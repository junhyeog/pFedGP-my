import argparse
import copy
import logging
import os
import uuid
from collections import OrderedDict, defaultdict
from pathlib import Path
from time import sleep

import numpy as np
import optuna
import torch
import torch.utils.data
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from experiments.backbone import CNNCifar, CNNTarget
from experiments.ood_generalization.clients import GenBaseClients

# from ..backbone import CNNCifar, CNNTarget
# from .clients import GenBaseClients
from pFedGP_my.Learner import pFedGPFullLearner
from utils import calc_metrics, calc_weighted_metrics, get_device, offset_client_classes, save_experiment, set_logger, set_seed, str2bool

parser = argparse.ArgumentParser(description="Personalized Federated Learning")

#############################
#       Dataset Args        #
#############################
parser.add_argument(
    "--data-name",
    type=str,
    default="cifar10",
    choices=["cifar10", "cifar100"],
)
parser.add_argument("--data-path", type=str, default="datafolder", help="dir path for CIFAR datafolder")
parser.add_argument("--num-clients", type=int, default=130, help="number of simulated clients")
parser.add_argument("--alpha", type=float, default=0.1, help="alpha param for diri distribution")
parser.add_argument("--alpha-gen", type=lambda s: [float(item.strip()) for item in s.split(",")], default="0.1,0.25,0.5,0.75,1.0", help="alpha on test")

##################################
#       Optimization args        #
##################################
parser.add_argument("--num-steps", type=int, default=1000)
parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"], help="learning rate")
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--inner-steps", type=int, default=5, help="number of inner steps")
parser.add_argument("--num-client-agg", type=int, default=10, help="number of kernels")
parser.add_argument("--num-novel-clients", type=int, default=30)

################################
#       Model Prop args        #
################################
parser.add_argument("--lr", type=float, default=5e-2, help="learning rate")
parser.add_argument("--wd", type=float, default=1e-3, help="weight decay")
parser.add_argument("--n-kernels", type=int, default=16, help="number of kernels")

parser.add_argument("--embed-dim", type=int, default=84)
parser.add_argument("--loss-scaler", default=1.0, type=float, help="multiplicative element to the loss function")
parser.add_argument("--kernel-function", type=str, default="RBFKernel", choices=["RBFKernel", "LinearKernel", "MaternKernel"], help="kernel function")
parser.add_argument("--objective", type=str, default="predictive_likelihood", choices=["predictive_likelihood", "marginal_likelihood"])
parser.add_argument("--predict-ratio", type=float, default=0.5, help="ratio of samples to make predictions for when using predictive_likelihood objective")
parser.add_argument("--num-gibbs-steps-train", type=int, default=5, help="number of sampling iterations")
parser.add_argument("--num-gibbs-draws-train", type=int, default=20, help="number of parallel gibbs chains")
parser.add_argument("--num-gibbs-steps-test", type=int, default=5, help="number of sampling iterations")
parser.add_argument("--num-gibbs-draws-test", type=int, default=30, help="number of parallel gibbs chains")
parser.add_argument("--outputscale", type=float, default=8.0, help="output scale")
parser.add_argument("--lengthscale", type=float, default=1.0, help="length scale")
parser.add_argument("--outputscale-increase", type=str, default="constant", choices=["constant", "increase", "decrease"], help="output scale increase/decrease/constant along tree")

#############################
#       General args        #
#############################
parser.add_argument("--gpus", type=str, default="0", help="gpu device ID")
parser.add_argument("--exp-name", type=str, default="", help="suffix for exp name")
parser.add_argument("--eval-every", type=int, default=100, help="eval every X selected steps")
parser.add_argument("--save-path", type=str, default="./output/pFedGP", help="dir path for output file")  # change
parser.add_argument("--seed", type=int, default=42, help="seed value")

parser.add_argument("--env", type=str, default="pfedgp", help="experiment environment")
parser.add_argument("--get-data-type", type=int, default=2)

parser.add_argument("--sampler", type=str, default="TPE")
parser.add_argument("--n_trials", type=int, default=4)
parser.add_argument("--adap_epochs", type=int, default=1)


def main(args, trial):
    # optuna params
    params = {
        "lr": trial.suggest_float("lr", 0.005, 1, step=0.005),
        "wd": trial.suggest_float("wd", 5e-4, 1e-3, step=1e-4),
        "inner_steps": trial.suggest_int("inner_steps", 1, 5, step=2),
        "adap_epochs": trial.suggest_int("adap_epochs", 1, 5, step=2),
        "num_gibbs_draws_train": trial.suggest_int("num_gibbs_draws_train", 20, 20, step=10),
        "num_gibbs_draws_test": trial.suggest_int("num_gibbs_draws_test", 30, 30, step=10),
    }
    # round float params
    for k, v in params.items():
        if isinstance(v, float):
            params[k] = round(v, 10)
    # update args
    vars(args).update(params)
    # check args
    print(f"[+] Args:")
    for k, v in vars(args).items():
        print(f" - {k:20}: {v}")
    # args.trial = trial
    # avoid duplicate sets
    for previous_trial in trial.study.trials:
        if previous_trial.state == optuna.trial.TrialState.COMPLETE and trial.params == previous_trial.params:
            print(f"[+] Duplicated trial: {trial.params}, return {previous_trial.values}")
            return previous_trial.values

    set_logger()
    set_seed(args.seed)

    device = get_device(cuda=int(args.gpus) >= 0, gpus=args.gpus)
    num_classes = 10 if args.data_name == "cifar10" else 100

    # exp_name = f'pFedGP-OOD-Gen_{args.data_name}_num_clients_{args.num_clients}_seed_{args.seed}_' \
    #            f'lr_{args.lr}_num_steps_{args.num_steps}_inner_steps_{args.inner_steps}_' \
    #            f'objective_{args.objective}_predict_ratio_{args.predict_ratio}' \
    #            f'_alpha_{args.alpha}_num_novel_{args.num_novel_clients}'

    # if args.exp_name != '':
    #     exp_name += '_' + args.exp_name

    args.uuid = str(uuid.uuid1())[:8]
    exp_name = f"env:{args.env}_s:{args.seed}_"
    exp_name += f"d:{args.data_name}_a:{args.alpha}_"
    exp_name += f"c:{args.num_clients},{args.num_client_agg},{args.num_novel_clients}_"
    exp_name += f"T:{args.num_steps}_is:{args.inner_steps}_ae:{args.adap_epochs}_"
    exp_name += f"lr:{args.lr}_bs:{args.batch_size}_"
    exp_name += f"optim:{args.optimizer}_wd:{args.wd}_"
    exp_name += f"gdt:{args.get_data_type}_"
    exp_name += f"obj:{args.objective}_"
    exp_name += f"ngd_train:{args.num_gibbs_draws_train}_"
    exp_name += f"ngd_test:{args.num_gibbs_draws_test}_"
    exp_name += f"{args.uuid}_{trial.number}"

    args.out_dir = (Path(args.save_path) / exp_name).as_posix()
    out_dir = save_experiment(args, None, return_out_dir=True, save_results=False)
    logging.warning(f"[+] out_dir: {out_dir}")
    writer = SummaryWriter(out_dir)

    @torch.no_grad()
    def eval_model(global_model, client_ids, GPs, clients, split, ratio=1):
        results = defaultdict(lambda: defaultdict(list))
        global_model.eval()
        sampled_clients = np.random.choice(client_ids, int(len(client_ids) * ratio), replace=False)
        pbar = tqdm(sampled_clients)
        # pbar = tqdm(client_ids)
        for client_id in pbar:
            is_first_iter = True
            running_loss, running_correct, running_samples = 0.0, 0.0, 0.0
            if split == "test":
                curr_data = clients.test_loaders[client_id]
            elif split == "val":
                curr_data = clients.val_loaders[client_id]
            else:
                curr_data = clients.train_loaders[client_id]

            GPs[client_id], label_map, X_train, Y_train = build_tree(clients, client_id)
            GPs[client_id].eval()

            for batch_count, batch in enumerate(curr_data):
                img, label = tuple(t.to(device) for t in batch)
                Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype, device=label.device)

                X_test = global_model(img)
                loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)

                running_loss += loss.item()
                running_correct += pred.argmax(1).eq(Y_test).sum().item()
                running_samples += len(Y_test)

                is_first_iter = False

            # erase tree (no need to save it)
            GPs[client_id].tree = None

            if running_samples > 0:
                results[client_id]["loss"] = running_loss / (batch_count + 1)
                results[client_id]["correct"] = running_correct
                results[client_id]["total"] = running_samples

        return results

    def eval_model_after_adap(init_global_model, client_ids, GPs, clients, split, ratio=1):
        results = defaultdict(lambda: defaultdict(list))
        sampled_clients = np.random.choice(client_ids, int(len(client_ids) * ratio), replace=False)
        pbar = tqdm(sampled_clients)
        ###
        train_avg_loss = 0
        num_samples = 0

        for j, client_id in enumerate(pbar):
            is_first_iter = True
            # * train
            curr_global_net = copy.deepcopy(init_global_model)
            curr_global_net.train()
            optimizer = get_optimizer(curr_global_net)

            # build tree at each step
            GPs[client_id], label_map, _, __ = build_tree(clients, client_id)
            GPs[client_id].train()

            for i in range(args.adap_epochs):
                # init optimizers
                optimizer.zero_grad()

                # With GP take all data
                for k, batch in enumerate(clients.train_loaders[client_id]):
                    batch = (t.to(device) for t in batch)
                    img, label = batch

                    z = curr_global_net(img)
                    X = torch.cat((X, z), dim=0) if k > 0 else z
                    Y = torch.cat((Y, label), dim=0) if k > 0 else label

                offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype, device=Y.device)

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

            # erase tree (no need to save it)
            GPs[client_id].tree = None

            # * test
            with torch.no_grad():
                global_model = curr_global_net
                global_model.eval()

                running_loss, running_correct, running_samples = 0.0, 0.0, 0.0
                if split == "test":
                    curr_data = clients.test_loaders[client_id]
                elif split == "val":
                    curr_data = clients.val_loaders[client_id]
                else:
                    curr_data = clients.train_loaders[client_id]

                GPs[client_id], label_map, X_train, Y_train = build_tree(clients, client_id)
                GPs[client_id].eval()

                for batch_count, batch in enumerate(curr_data):
                    img, label = tuple(t.to(device) for t in batch)
                    Y_test = torch.tensor([label_map[l.item()] for l in label], dtype=label.dtype, device=label.device)

                    X_test = global_model(img)
                    loss, pred = GPs[client_id].forward_eval(X_train, Y_train, X_test, Y_test, is_first_iter)

                    running_loss += loss.item()
                    running_correct += pred.argmax(1).eq(Y_test).sum().item()
                    running_samples += len(Y_test)

                    is_first_iter = False

                # erase tree (no need to save it)
                GPs[client_id].tree = None

                if running_samples > 0:
                    results[client_id]["loss"] = running_loss / (batch_count + 1)
                    results[client_id]["correct"] = running_correct
                    results[client_id]["total"] = running_samples

                step_iter.set_description(
                    f"Test after adaptation, client: {client_id}, Loss: {results[client_id]['loss']}, Acc: {results[client_id]['correct'] / results[client_id]['total']:.4f} ({results[client_id]['correct']}/{results[client_id]['total']}))"
                )

        train_avg_loss /= num_samples
        writer.add_scalar("adap/loss", train_avg_loss, step)
        return results

    ###############################
    # init net and GP #
    ###############################
    def client_counts(num_clients, split="train"):
        client_num_classes = {}
        for client_id in range(num_clients):
            if split == "test":
                curr_data = clients.test_loaders[client_id]
            elif split == "val":
                curr_data = clients.val_loaders[client_id]
            else:
                curr_data = clients.train_loaders[client_id]

            for i, batch in enumerate(curr_data):
                img, label = tuple(t.to(device) for t in batch)
                all_labels = label if i == 0 else torch.cat((all_labels, label))

            client_labels, client_counts = torch.unique(all_labels, return_counts=True)
            client_num_classes[client_id] = client_labels.shape[0]
        return client_num_classes

    def client_counts_data(num_clients, split="train"):
        client_num_data = {}
        for client_id in range(num_clients):
            if split == "test":
                curr_data = clients.test_loaders[client_id]
            elif split == "val":
                curr_data = clients.val_loaders[client_id]
            else:
                curr_data = clients.train_loaders[client_id]

            cnt = 0
            for i, batch in enumerate(curr_data):
                cnt += batch[0].shape[0]

            client_num_data[client_id] = cnt
        return client_num_data

    clients = GenBaseClients(args.data_name, args.data_path, args.num_clients, n_gen_clients=args.num_novel_clients, alpha=args.alpha, batch_size=args.batch_size, args=args)
    client_num_classes = client_counts(args.num_clients)
    client_datas_size_train = client_counts_data(args.num_clients, "train")
    client_datas_size_val = client_counts_data(args.num_clients, "val")
    client_datas_size_test = client_counts_data(args.num_clients, "test")

    logging.warning(f"[+] (train) Client num classes: \n{client_num_classes}")
    logging.warning(f"[+] (train) Client data size: \n{client_datas_size_train}")
    logging.warning(f"[+] (val) Client data size: \n{client_datas_size_val}")
    logging.warning(f"[+] (test) Client data size: \n{client_datas_size_test}")

    # NN
    if "pfedgp" in args.env:
        net = CNNTarget(n_kernels=args.n_kernels, embedding_dim=args.embed_dim)
        logging.warning(f"[+] Using CNNTarget(n_kernels={args.n_kernels}, embedding_dim={args.embed_dim})")
    elif "bmfl" in args.env:
        net = CNNCifar(embedding_dim=args.embed_dim)
        logging.warning(f"[+] Using CNNCifar(embedding_dim={args.embed_dim})")
    net = net.to(device)

    GPs = torch.nn.ModuleList([])
    for client_id in range(args.num_clients):
        GPs.append(pFedGPFullLearner(args, client_num_classes[client_id]))  # GP instances

    def get_optimizer(network):
        return (
            torch.optim.SGD(network.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
            if args.optimizer == "sgd"
            else torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.wd)
        )

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
        offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype, device=Y.device)

        GPs[client_id].build_base_tree(X, offset_labels)  # build tree
        return GPs[client_id], label_map, X, offset_labels

    criteria = torch.nn.CrossEntropyLoss()

    ################
    # init metrics #
    ################
    last_eval = -1
    step_iter = trange(args.num_steps)
    results = defaultdict(list)

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

                offset_labels = torch.tensor([label_map[l.item()] for l in Y], dtype=Y.dtype, device=Y.device)

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

            for n, p in curr_global_net.named_parameters():
                params[n] += p.data
            # erase tree (no need to save it)
            GPs[client_id].tree = None

        train_avg_loss /= num_samples
        writer.add_scalar("train/loss", train_avg_loss, step)
        step_iter.set_description(f"Train, Step: {step+1}, Loss: {train_avg_loss}")

        # average parameters
        for n, p in params.items():
            params[n] = p / args.num_client_agg
        # update new parameters
        net.load_state_dict(params)

        if (step + 1) % args.eval_every == 0 or (step + 1) == args.num_steps:
            ratio = 0.3

            # ! >>> fixed
            train_results = eval_model(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="train", ratio=1 if (step + 1) == args.num_steps else ratio)
            train_avg_loss, train_avg_acc = calc_metrics(train_results)
            train_avg_loss_weighted, train_avg_acc_weighted = calc_weighted_metrics(train_results, client_datas_size_train)
            logging.info(f"[+] (train, ratio={ratio}) Step: {step + 1}, AVG Loss: {train_avg_loss:.4f},  AVG Acc train: {train_avg_acc:.4f}")
            logging.info(f"[+] (train, ratio={ratio}) Step: {step + 1}, Weighted Loss: {train_avg_loss_weighted:.4f},  Weighted Acc train: {train_avg_acc_weighted:.4f}")

            writer.add_scalar(f"train_{ratio}/loss", train_avg_loss, step)
            writer.add_scalar(f"train_{ratio}/acc", train_avg_acc, step)
            writer.add_scalar(f"train_{ratio}/loss_weighted", train_avg_loss_weighted, step)
            writer.add_scalar(f"train_{ratio}/acc_weighted", train_avg_acc_weighted, step)
            ### ! <<<

            val_results = eval_model_after_adap(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="test", ratio=1 if (step + 1) == args.num_steps else ratio)
            val_avg_loss, val_avg_acc = calc_metrics(val_results)
            val_avg_loss_weighted, val_avg_acc_weighted = calc_weighted_metrics(val_results, client_datas_size_val)
            logging.info(f"[+] (val, ratio={ratio}) Step: {step + 1}, AVG Loss: {val_avg_loss:.4f},  AVG Acc Val: {val_avg_acc:.4f}")
            logging.info(f"[+] (val, ratio={ratio}) Step: {step + 1}, Weighted Loss: {val_avg_loss_weighted:.4f},  Weighted Acc Val: {val_avg_acc_weighted:.4f}")

            writer.add_scalar(f"val_{ratio}/loss", val_avg_loss, step)
            writer.add_scalar(f"val_{ratio}/acc", val_avg_acc, step)
            writer.add_scalar(f"val_{ratio}/loss_weighted", val_avg_loss_weighted, step)
            writer.add_scalar(f"val_{ratio}/acc_weighted", val_avg_acc_weighted, step)

            ### ! fixed >>> test ood user during training
            ood_results = eval_model_after_adap(net, range(args.num_novel_clients), GPs, clients, split="test")
            avg_ood_loss, avg_ood_acc = calc_metrics(ood_results)
            avg_ood_loss_weighted, avg_ood_acc_weighted = calc_weighted_metrics(ood_results, client_datas_size_test)

            logging.info(f"[+] (ood, alpha={args.alpha}) ood loss: {avg_ood_loss}, ood acc: {avg_ood_acc}")
            logging.info(f"[+] (ood, alpha={args.alpha}) ood loss weighted: {avg_ood_loss_weighted}, ood acc weighted: {avg_ood_acc_weighted}")
            writer.add_scalar(f"ood_{args.alpha}/loss", avg_ood_loss, step)
            writer.add_scalar(f"ood_{args.alpha}/acc", avg_ood_acc, step)
            writer.add_scalar(f"ood_{args.alpha}/loss_weighted", avg_ood_loss_weighted, step)
            writer.add_scalar(f"ood_{args.alpha}/acc_weighted", avg_ood_acc_weighted, step)
            ### ! <<<

    logging.info(f"\n[+] (train) loss: {train_avg_loss:.4f}")

    # ! save
    ckpt_path = os.path.join(out_dir, f"ckpt.pth")
    torch.save({"args": args, "net": net.state_dict()}, ckpt_path)
    logging.warning(f"[+] Saved checkpoint to {ckpt_path}")

    # # ! final test
    # test_results = eval_model(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="test")
    # avg_test_loss, avg_test_acc = calc_metrics(test_results)
    # avg_test_loss_weighted, avg_test_acc_weighted = calc_weighted_metrics(test_results, client_datas_size_test)

    # logging.info(f"\n(test) Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
    # logging.info(f"\n(test) Test Loss Weighted: {avg_test_loss_weighted:.4f}, Test Acc Weighted: {avg_test_acc_weighted:.4f}")

    # results["test_loss"].append(avg_test_loss)
    # results["test_acc"].append(avg_test_acc)
    # results["test_loss_weighted"].append(avg_test_loss_weighted)
    # results["test_acc_weighted"].append(avg_test_acc_weighted)
    # writer.add_scalar("test/loss", avg_test_loss, step)
    # writer.add_scalar("test/acc", avg_test_acc, step)
    # writer.add_scalar("test/loss_weighted", avg_test_loss_weighted, step)
    # writer.add_scalar("test/acc_weighted", avg_test_acc_weighted, step)

    # #########################
    # # generalization to ood #
    # #########################
    # args.alpha_gen = [args.alpha]
    # for alpha_gen in args.alpha_gen:
    #     clients = GenBaseClients(
    #         data_name=args.data_name,
    #         data_path=args.data_path,
    #         n_clients=args.num_clients,
    #         n_gen_clients=args.num_novel_clients,
    #         alpha=alpha_gen,
    #         batch_size=args.batch_size,
    #         args=args,
    #     )

    #     client_num_classes = client_counts(args.num_clients)
    #     client_datas_size_test = client_counts_data(args.num_clients, "test")
    #     logging.info(f"[+] (ood, alpha={alpha_gen}) Client num classes: \n{client_num_classes}")
    #     logging.info(f"[+] (ood, alpha={alpha_gen}) Client datas size test: \n{client_datas_size_test}")

    #     # GPs
    #     GPs = torch.nn.ModuleList([])
    #     for client_id in range(args.num_clients):
    #         # GP instance
    #         GPs.append(pFedGPFullLearner(args, client_num_classes[client_id]))

    #     ood_results = eval_model(net, range(args.num_novel_clients), GPs, clients, split="test")
    #     avg_ood_loss, avg_ood_acc = calc_metrics(ood_results)
    #     avg_ood_loss_weighted, avg_ood_acc_weighted = calc_weighted_metrics(ood_results, client_datas_size_test)

    #     logging.info(f"[+] (final_ood, alpha={alpha_gen}) ood loss: {avg_ood_loss}, ood acc: {avg_ood_acc}")
    #     logging.info(f"[+] (final_ood, alpha={alpha_gen}) ood loss weighted: {avg_ood_loss_weighted}, ood acc weighted: {avg_ood_acc_weighted}")
    #     writer.add_scalar(f"final_ood_{alpha_gen}/loss", avg_ood_loss, step)
    #     writer.add_scalar(f"final_ood_{alpha_gen}/acc", avg_ood_acc, step)
    #     writer.add_scalar(f"final_ood_{alpha_gen}/loss_weighted", avg_ood_loss_weighted, step)
    #     writer.add_scalar(f"final_ood_{alpha_gen}/acc_weighted", avg_ood_acc_weighted, step)
    #     writer.flush()

    #     # ! test
    #     final_test_results = eval_model(net, range(args.num_novel_clients, args.num_clients), GPs, clients, split="test")
    #     avg_test_loss, avg_test_acc = calc_metrics(final_test_results)
    #     avg_test_loss_weighted, avg_test_acc_weighted = calc_weighted_metrics(final_test_results, client_datas_size_test)

    #     logging.info(f"\n(final_test) Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
    #     logging.info(f"\n(final_test) Test Loss Weighted: {avg_test_loss_weighted:.4f}, Test Acc Weighted: {avg_test_acc_weighted:.4f}")

    #     writer.add_scalar("final_test/loss", avg_test_loss, step)
    #     writer.add_scalar("final_test/acc", avg_test_acc, step)
    #     writer.add_scalar("final_test/loss_weighted", avg_test_loss_weighted, step)
    #     writer.add_scalar("final_test/acc_weighted", avg_test_acc_weighted, step)
    #     writer.flush()

    return val_avg_acc_weighted, avg_ood_acc_weighted


### ! optuna
if __name__ == "__main__":
    secs = getattr(os, "getppid", lambda: 0)() % 3
    print(f"Sleeping for {secs} seconds...")
    sleep(secs)

    args = parser.parse_args()

    args.save_path = f"output/{args.exp_name}"

    # # ##### OPTUNA #######
    os.makedirs(f"studys/{args.exp_name}", exist_ok=True)
    study_name = f"env_{args.env}_seed_{args.seed}_d_{args.data_name}_alpha_{args.alpha}_gdt_{args.get_data_type}"
    args.study_name = study_name

    if args.sampler == "TPE":
        sampler = optuna.samplers.TPESampler()
    elif args.sampler == "RAND":
        sampler = optuna.samplers.RandomSampler()

    # from optuna.storages import RetryFailedTrialCallback
    storage = optuna.storages.RDBStorage(
        url=f"sqlite:///studys/{args.exp_name}/{study_name}.db",
        # url=f"sqlite:///studys/11.db",
        # url=f"sqlite://///gallery_moma/junhyeog.yun/Git/pFedGP-my/studys/{study_name}.db",
        # engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
        heartbeat_interval=60,
        grace_period=120,
        # failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        directions=["maximize", "maximize"],
        sampler=sampler,
    )

    main_obj = lambda t: main(args, trial=t)
    study.optimize(main_obj, n_trials=args.n_trials, show_progress_bar=True)


### !
