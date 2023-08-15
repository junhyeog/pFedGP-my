import argparse
import copy
import logging
import os
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from experiments.backbone import CNNCifar, CNNTarget
from experiments.ood_generalization.clients import GenBaseClients
from pFedGP_my.Learner import pFedGPFullLearner
from utils import (calc_metrics, calc_weighted_metrics, get_device,
                   offset_client_classes, save_experiment, set_logger,
                   set_seed, str2bool)

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

#############################
#       Dataset Args        #
#############################
parser.add_argument(
    "--data-name", type=str, default="cifar10", choices=['cifar10'],
)
parser.add_argument("--data-path", type=str, default="../datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--num-clients", type=int, default=100, help="number of simulated clients")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha param for diri distribution")
parser.add_argument("--alpha-gen", type=lambda s: [float(item.strip()) for item in s.split(',')],
                    default='0.1,0.25,0.5,0.75,1.0',
                    help='alpha on test')

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--optimizer", type=str, default='sgd', choices=['adam', 'sgd'], help="learning rate")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--inner-steps", type=int, default=1, help="number of inner steps")
parser.add_argument("--num-client-agg", type=int, default=5, help="number of kernels")
parser.add_argument("--num-novel-clients", type=int, default=10)

################################
#       Model Prop args        #
################################
parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

parser.add_argument('--embed-dim', type=int, default=84)
parser.add_argument('--loss-scaler', default=1., type=float, help='multiplicative element to the loss function')
parser.add_argument('--kernel-function', type=str, default='RBFKernel',
                    choices=['RBFKernel', 'LinearKernel', 'MaternKernel'],
                    help='kernel function')
parser.add_argument('--objective', type=str, default='predictive_likelihood',
                    choices=['predictive_likelihood', 'marginal_likelihood'])
parser.add_argument('--predict-ratio', type=float, default=0.5,
                    help='ratio of samples to make predictions for when using predictive_likelihood objective')
parser.add_argument('--num-gibbs-steps-train', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-train', type=int, default=20, help='number of parallel gibbs chains')
parser.add_argument('--num-gibbs-steps-test', type=int, default=5, help='number of sampling iterations')
parser.add_argument('--num-gibbs-draws-test', type=int, default=30, help='number of parallel gibbs chains')
parser.add_argument('--outputscale', type=float, default=8., help='output scale')
parser.add_argument('--lengthscale', type=float, default=1., help='length scale')
parser.add_argument('--outputscale-increase', type=str, default='constant',
                    choices=['constant', 'increase', 'decrease'],
                    help='output scale increase/decrease/constant along tree')

#############################
#       General args        #
#############################
parser.add_argument("--gpus", type=str, default='0', help="gpu device ID")
parser.add_argument("--exp-name", type=str, default='', help="suffix for exp name")
parser.add_argument("--eval-every", type=int, default=25, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")  # change
parser.add_argument("--seed", type=int, default=42, help="seed value")

parser.add_argument("--env", type=str, default='pfedgp', choices=['pfedgp', 'bmfl'], help="experiment environment")
parser.add_argument("--get-data-type", type=int, default=1)
args = parser.parse_args()
        
set_logger()
set_seed(args.seed)

device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
num_classes = 10 if args.data_name == 'cifar10' else 100


exp_name = f'{args.exp_name}_env:{args.env}_seed:{args.seed}_'
exp_name += f'd:{args.data_name}_alpha:{args.alpha}_'
exp_name += f'clients:{args.num_clients},{args.num_client_agg},{args.num_novel_clients}_'
exp_name += f'T:{args.num_steps}_is:{args.inner_steps}_'
exp_name += f'lr:{args.lr}_bs:{args.batch_size}_'
exp_name += f'optim:{args.optimizer}_wd:{args.wd}_'
exp_name += f'gdt:{args.get_data_type}_'

logging.info(str(args))
args.out_dir = (Path(args.save_path) / exp_name).as_posix()
out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
logging.info(out_dir)
writer = SummaryWriter(out_dir)

@torch.no_grad()
def eval_model(global_model, client_ids, GPs, clients, split):
    results = defaultdict(lambda: defaultdict(list))
    global_model.eval()

    for client_id in client_ids:
        is_first_iter = True
        running_loss, running_correct, running_samples = 0., 0., 0.
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        GPs[client_id], label_map, X_train, Y_train = build_tree(clients, client_id)
        GPs[client_id].eval()

        for batch_count, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype,
                                         device=label.device)

            X_test = global_model(img)
            loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)

            running_loss += loss.item()
            running_correct += pred.argmax(1).eq(Y_test).sum().item()
            running_samples += len(Y_test)

            is_first_iter = False

        # erase tree (no need to save it)
        GPs[client_id].tree = None

        if running_samples > 0:
            results[client_id]['loss'] = running_loss / (batch_count + 1)
            results[client_id]['correct'] = running_correct
            results[client_id]['total'] = running_samples

    return results

###############################
# init net and GP #
###############################
def client_counts(num_clients, split='train'):
    client_num_classes = {}
    for client_id in range(num_clients):
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        for i, batch in enumerate(curr_data):
            img, label = tuple(t.to(device) for t in batch)
            all_labels = label if i == 0 else torch.cat((all_labels, label))

        client_labels, client_counts = torch.unique(all_labels, return_counts=True)
        client_num_classes[client_id] = client_labels.shape[0]
    return client_num_classes

def client_counts_data(num_clients, split='train'):
    client_num_data = {}
    for client_id in range(num_clients):
        if split == 'test':
            curr_data = clients.test_loaders[client_id]
        elif split == 'val':
            curr_data = clients.val_loaders[client_id]
        else:
            curr_data = clients.train_loaders[client_id]

        cnt = 0
        for i, batch in enumerate(curr_data):
            cnt += batch[0].shape[0]

        client_num_data[client_id] = cnt
    return client_num_data

clients = GenBaseClients(args.data_name, args.data_path, args.num_clients,
                       n_gen_clients=args.num_novel_clients,
                       alpha=args.alpha,
                       batch_size=args.batch_size,
                       args=args)
client_num_classes = client_counts(args.num_clients)
client_datas_size_train= client_counts_data(args.num_clients, 'train')
client_datas_size_val= client_counts_data(args.num_clients, 'val')
client_datas_size_test= client_counts_data(args.num_clients, 'test')

logging.info(f"[+] (train) Client num classes: \n{client_num_classes}")

# NN
if 'pfedgp' in args.env:
    net = CNNTarget(n_kernels=args.n_kernels, embedding_dim=args.embed_dim)
    logging.info(f'[+] Using CNNTarget(n_kernels={args.n_kernels}, embedding_dim={args.embed_dim})')
elif 'bmfl' in args.env:
    net = CNNCifar(embedding_dim=args.embed_dim)
    logging.info(f'[+] Using CNNCifar(embedding_dim={args.embed_dim})')
net = net.to(device)


GPs = torch.nn.ModuleList([])
for client_id in range(args.num_clients):
    GPs.append(pFedGPFullLearner(args, client_num_classes[client_id]))  # GP instances


def get_optimizer(network):
    return torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9) \
           if args.optimizer == 'sgd' else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)

@torch.no_grad()
def build_tree(clients, client_id):
    """
    Build GP tree per client
    :return: List of GPs
    """
    for k, batch in enumerate(clients.train_loaders[client_id]):
        batch = (t.to(device) for t in batch)
        train_data, clf_labels = batch

        z = net(train_data)
        X = torch.cat((X, z), dim=0) if k > 0 else z
        Y = torch.cat((Y, clf_labels), dim=0) if k > 0 else clf_labels

    # build label map
    client_labels, client_indices = torch.sort(torch.unique(Y))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                 device=Y.device)

    GPs[client_id].build_base_tree(X, offset_labels)  # build tree
    return GPs[client_id], label_map, X, offset_labels


criteria = torch.nn.CrossEntropyLoss()

################
# init metrics #
################
last_eval = -1
best_step = -1
best_acc = -1
test_best_based_on_step, test_best_min_based_on_step = -1, -1
test_best_max_based_on_step, test_best_std_based_on_step = -1, -1
step_iter = trange(args.num_steps)
results = defaultdict(list)

best_model = copy.deepcopy(net)
best_labels_vs_preds_val = None
best_val_loss = -1

for step in step_iter:

    # print tree stats every 100 epochs
    to_print = True if step % 100 == 0 else False

    # select several clients
    client_ids = np.random.choice(range(args.num_novel_clients, args.num_clients), size=args.num_client_agg, replace=False)

    # initialize global model params
    params = OrderedDict()
    for n, p in net.named_parameters():
        params[n] = torch.zeros_like(p.data)

    # iterate over each client
    train_avg_loss = 0
    num_samples = 0

    for j, client_id in enumerate(client_ids):
        curr_global_net = copy.deepcopy(net)
        curr_global_net.train()
        optimizer = get_optimizer(curr_global_net)

        # build tree at each step
        GPs[client_id], label_map, _, __ = build_tree(clients, client_id)
        GPs[client_id].train()

        for i in range(args.inner_steps):

            # init optimizers
            optimizer.zero_grad()

            # With GP take all data
            for k, batch in enumerate(clients.train_loaders[client_id]):
                batch = (t.to(device) for t in batch)
                img, label = batch

                z = curr_global_net(img)
                X = torch.cat((X, z), dim=0) if k > 0 else z
                Y = torch.cat((Y, label), dim=0) if k > 0 else label

            offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype,
                                         device=Y.device)

            loss = GPs[client_id](X, offset_labels, to_print=to_print)
            loss *= args.loss_scaler

            # propagate loss
            #### >>> original code
            # loss.backward()
            # torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
            # optimizer.step()
            ### <<<
            ### ! >>> fixed
            if isinstance(loss, float):
                loss = torch.tensor(loss)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(curr_global_net.parameters(), 50)
                optimizer.step()
            ### ! <<<

            train_avg_loss += loss.item() * offset_labels.shape[0]
            num_samples += offset_labels.shape[0]

            step_iter.set_description(
                f"Step: {step+1}, client: {client_id}, Inner Step: {i}, Loss: {loss.item()}"
            )

        for n, p in curr_global_net.named_parameters():
            params[n] += p.data
        # erase tree (no need to save it)
        GPs[client_id].tree = None

    train_avg_loss /= num_samples
    writer.add_scalar("train/loss", train_avg_loss, step)

    # average parameters
    for n, p in params.items():
        params[n] = p / args.num_client_agg
    # update new parameters
    net.load_state_dict(params)

    if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
        ratio=1
        val_results = eval_model(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="val")
        val_avg_loss, val_avg_acc = calc_metrics(val_results)
        try:
            val_avg_loss_weighted, val_avg_acc_weighted = calc_weighted_metrics(val_results, client_datas_size_val)
        except:
            ###
            import ipdb
            ipdb.set_trace(context=5)
            ###
        logging.info(f"[+] (val, ratio={ratio}) Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")
        logging.info(f"[+] (val, ratio={ratio}) Step: {step + 1}, Weighted Loss: {val_avg_loss_weighted:.4f},  Weighted Acc Val: {val_avg_acc_weighted:.4f}")

        if best_acc < val_avg_acc:
            best_val_loss = val_avg_loss
            best_acc = val_avg_acc
            best_step = step
            best_model = copy.deepcopy(net)

        results['val_avg_loss'].append(val_avg_loss)
        results['val_avg_acc'].append(val_avg_acc)
        results['val_avg_loss_weighted'].append(val_avg_loss_weighted)
        results['val_avg_acc_weighted'].append(val_avg_acc_weighted)
        writer.add_scalar(f"val_{ratio}/loss", val_avg_loss, step)
        writer.add_scalar(f"val_{ratio}/acc", val_avg_acc, step)
        writer.add_scalar(f"val_{ratio}/loss_weighted", val_avg_loss_weighted, step)
        writer.add_scalar(f"val_{ratio}/acc_weighted", val_avg_acc_weighted, step)


logging.info(f"\n[+] (train) loss: {train_avg_loss:.4f}")

# ! save
ckpt_path = os.path.join(out_dir, f"ckpt.pth")
torch.save({"args": args, "net":net.state_dict()}, ckpt_path)
logging.info(f"[+] Saved checkpoint to {ckpt_path}")

# ! test
# net = best_model # !!!!!!!!!!!!!!!!!!!!!
test_results = eval_model(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="test")
avg_test_loss, avg_test_acc = calc_metrics(test_results)
avg_test_loss_weighted, avg_test_acc_weighted = calc_weighted_metrics(test_results, client_datas_size_test)

logging.info(f"\n(test) Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
logging.info(f"\n(test) Test Loss Weighted: {avg_test_loss_weighted:.4f}, Test Acc Weighted: {avg_test_acc_weighted:.4f}")

results['test_loss'].append(avg_test_loss)
results['test_acc'].append(avg_test_acc)
results['test_loss_weighted'].append(avg_test_loss_weighted)
results['test_acc_weighted'].append(avg_test_acc_weighted)
writer.add_scalar("test/loss", avg_test_loss, step)
writer.add_scalar("test/acc", avg_test_acc, step)
writer.add_scalar("test/loss_weighted", avg_test_loss_weighted, step)
writer.add_scalar("test/acc_weighted", avg_test_acc_weighted, step)



#########################
# generalization to ood #
#########################
args.alpha_gen = [args.alpha]
for alpha_gen in args.alpha_gen:
    clients = GenBaseClients(data_name=args.data_name, data_path=args.data_path, n_clients=args.num_clients,
                           n_gen_clients=args.num_novel_clients,
                           alpha=alpha_gen,
                           batch_size=args.batch_size,
                           args=args)

    client_num_classes = client_counts(args.num_clients)
    logging.info(f"[+] (ood, alpha={alpha_gen}) Client num classes: \n{client_num_classes}")

    # GPs
    GPs = torch.nn.ModuleList([])
    for client_id in range(args.num_clients):
        # GP instance
        GPs.append(pFedGPFullLearner(args, client_num_classes[client_id]))

    test_results = eval_model(net, range(args.num_novel_clients), GPs, clients, split="test")
    avg_test_loss, avg_test_acc = calc_metrics(test_results)
    avg_test_loss_weighted, avg_test_acc_weighted = calc_weighted_metrics(test_results, client_datas_size_test)

    logging.info(f"[+] (final_ood, alpha={alpha_gen}) ood loss: {avg_test_loss}, ood acc: {avg_test_acc}")
    logging.info(f"[+] (final_ood, alpha={alpha_gen}) ood loss weighted: {avg_test_loss_weighted}, ood acc weighted: {avg_test_acc_weighted}")
    writer.add_scalar(f"final_ood_{alpha_gen}/loss", avg_test_loss, step)
    writer.add_scalar(f"final_ood_{alpha_gen}/acc", avg_test_acc, step)
    writer.add_scalar(f"final_ood_{alpha_gen}/loss_weighted", avg_test_loss_weighted, step)
    writer.add_scalar(f"final_ood_{alpha_gen}/acc_weighted", avg_test_acc_weighted, step)
    writer.flush()