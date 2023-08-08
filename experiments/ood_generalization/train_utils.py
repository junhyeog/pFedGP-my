import copy
import logging
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

# from utils.femnist import FEMNIST
#from models.Nets import CNNCifar, MobileNetCifar
#from models.ResNet import ResNet18, ResNet50

trans_mnist = transforms.Compose([
    transforms.ToTensor(), # TODO: channel is 1
    transforms.Normalize((0.1307,), (0.3081,)),
])
trans_emnist = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])
trans_celeba = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
trans_cifar10_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trans_cifar10_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
trans_cifar100_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])
trans_cifar100_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
])


def add_rand_transform(args, transform):
    # copt transform
    transform = deepcopy(transform)
    # find index of ToTensor
    to_tensor_idx = [i for i, t in enumerate(transform.transforms) if isinstance(t, transforms.ToTensor)][0]
    interpolationMode = transforms.InterpolationMode.BICUBIC # NEAREST, BILINEAR, BICUBIC
    randaug = transforms.RandAugment(args.ra_n, args.ra_m, interpolation = interpolationMode)
    transform.transforms.insert(to_tensor_idx, randaug)
    return transform

def update_transform(args, transform):
    if args.ra:
        transform = add_rand_transform(args, transform)
        print("-"*50)
        print("Add RandAugment to transform")
        print(transform)
        print("-"*50)
    return transform

# >>> data split from pFL-Bench
# https://github.com/alibaba/FederatedScope
def _split_according_to_prior(label, client_num, prior):  # 각 class에 대해서 prior와 같은 비율로 각 client에게 나눔
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))  # 각 client의 class별 개수
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)  # 각 class당 data 수 -> shape = (classes,)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]  # return이 tuple이라 [0]을 붙여줌
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] * len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):  # ceil이니까 nums_k가 더 많을 수 있으므로 그만큼을 랜덤하게 빼줌
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, np.cumsum(nums_k)[:-1]))]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label, client_num, alpha, min_size=1, prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py
    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError("Only support single-label tasks!")

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f"The number of sample should be " f"greater than" f" {client_num * min_size}."
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            if client_num <= 400:
                idx_slice = [idx_j + idx.tolist() for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))]
            else:
                idx_k_slice = [idx.tolist() for idx in np.split(idx_k, prop)]
                idxs = np.arange(len(idx_k_slice))
                np.random.shuffle(idxs)
                idx_slice = [idx_j + idx_k_slice[idx] for idx_j, idx in zip(idx_slice, idxs)]

            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])

    dict_users = {client_idx: np.array(idx_slice[client_idx]) for client_idx in range(client_num)}
    return dict_users # idx_slice

# <<< data split from pFL-Bench
def get_data(dataset, num_users, ood_users, alpha):
    # type 0:
    # remove_test_only = False
    # remove_train_only = True
    # move_data = False
    # copy_data = False

    ### error table: default setting, cifar100_5.0, cifar100_0.5, cifar10_0.1 (before fix train_test_split method)
    # remove_test_only: O, remove_train_only: O -> X, X, X, X
    # remove_test_only: O, remove_train_only: X -> O, X, X, X
    # remove_test_only: X, remove_train_only: O -> X, X, X, X
    # remove_test_only: X, remove_train_only: X -> ?, ?, ?, ?
    
    # move_data, k=1: -> X, X, X, X 
    # move_data, k=2: -> X, X, X, X 
    # move_data, k=3: -> ?, ?, ?, ?
    # move_data, k=4: -> X, X, X, X 

    # type 1:
    remove_test_only = False
    remove_train_only = False
    move_data = True
    copy_data = False
    k = 1

    ### after fix train_test_split method 
    # move_data, k=1: -> O, O, O, O
    # default setting: train 50148, test 9796
    # cifar100_5.0   : train 50002, test 9984
    # cifar100_0.5   : train 50581, test 8419 
    # cifar10_0.1    : train 50229, test 9703
    
    # type 2:
    # remove_test_only = False
    # remove_train_only = False
    # move_data = False
    # copy_data = True

    total_users = num_users + ood_users
    if dataset == 'cifar10':
        dataset_train = datasets.CIFAR10('data/cifar10', train=True, download=True, transform=trans_cifar10_train)
        dataset_test = datasets.CIFAR10('data/cifar10', train=False, download=True, transform=trans_cifar10_val)
    elif dataset == 'cifar100':
        dataset_train = datasets.CIFAR100('data/cifar100', train=True, download=True, transform=trans_cifar100_train)
        dataset_test = datasets.CIFAR100('data/cifar100', train=False, download=True, transform=trans_cifar100_val)
    else:
        exit('Error: unrecognized dataset')

    targets_train = np.array(dataset_train.targets)
    targets_test = np.array(dataset_test.targets)

    dict_users_train = dirichlet_distribution_noniid_slice(targets_train, total_users, alpha)
    dict_users_test = dirichlet_distribution_noniid_slice(targets_test, total_users, alpha)

    logging.info(f"[+] total original data size: train {sum([len(dict_users_train[user]) for user in range(total_users)])}, test {sum([len(dict_users_test[user]) for user in range(total_users)])}")
    
    if remove_test_only:
        # remove test only classes from test set
        for user in range(total_users):
            user_targets_train = targets_train[dict_users_train[user]]
            user_targets_test = targets_test[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            test_only = np.setdiff1d(test_classes, train_classes, assume_unique=True)

            for test_only_class in test_only:
                idxs = np.array([targets_test[idx] == test_only_class for idx in dict_users_test[user]])
                dict_users_test[user] = dict_users_test[user][~idxs]
                
            # check if there is any class only in train or test set
            user_targets_train = np.array(dataset_train.targets)[dict_users_train[user]]
            user_targets_test = np.array(dataset_test.targets)[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            # check if there is any class that is in test and not in train set
            for c in test_classes:
                if c not in train_classes:
                    logging.info(f"[!] test class {c} is not in train set")
                    raise ValueError(f"[!] test class {c} is not in train set")

        logging.info(f"[+] total data size after remove test only classes: train {sum([len(dict_users_train[user]) for user in range(total_users)])}, test {sum([len(dict_users_test[user]) for user in range(total_users)])}")
    
    if remove_train_only:
        # remove train only classes from train set
        for user in range(total_users):
            user_targets_train = targets_train[dict_users_train[user]]
            user_targets_test = targets_test[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            train_only = np.setdiff1d(train_classes, test_classes, assume_unique=True)

            for train_only_class in train_only:
                idxs = np.array([targets_train[idx] == train_only_class for idx in dict_users_train[user]])
                dict_users_train[user] = dict_users_train[user][~idxs]

            # check if there is any class only in train or test set
            user_targets_train = np.array(dataset_train.targets)[dict_users_train[user]]
            user_targets_test = np.array(dataset_test.targets)[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            # check if there is any class that is in train and not in test set
            for c in train_classes:
                if c not in test_classes:
                    logging.info(f"[!] train class {c} is not in test set")
                    raise ValueError(f"[!] train class {c} is not in test set")
        logging.info(f"[+] total data size after remove train only classes: train {sum([len(dict_users_train[user]) for user in range(total_users)])}, test {sum([len(dict_users_test[user]) for user in range(total_users)])}")

    if move_data:
        # 1. merge train and test set into one dataset (merged_dataset)
        # 2. update dict_users_train and dict_users_test for merged_dataset
        # 3. for each user, find test only classes
        # 4. for each test only class, find the test data of this class
        # 5. if the test data of this class is only one, remove it from test set
        # 6. else, move one random data of this class from test set to train set

        merged_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
        merged_targets = np.concatenate((targets_train, targets_test), axis=0)
        merged_dataset.targets = merged_targets
        dict_users_test_orig = copy.deepcopy(dict_users_test)
        for user_idx in range(total_users):
            dict_users_test[user_idx] = dict_users_test[user_idx] + len(dataset_train)

        # check merged dataset
        for user_idx in range(total_users):
            dict_user_orig = dict_users_test_orig[user_idx]
            dict_user = dict_users_test[user_idx]
            for idx_orig, idx in zip(dict_user_orig, dict_user):
                assert torch.allclose(dataset_test[idx_orig][0], merged_dataset[idx][0])
                assert dataset_test[idx_orig][1] == merged_dataset[idx][1]

        for user_idx in range(total_users):
            user_targets_train = merged_targets[dict_users_train[user_idx]]
            user_targets_test = merged_targets[dict_users_test[user_idx]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            test_only = np.setdiff1d(test_classes, train_classes, assume_unique=True)
            for test_only_class in test_only:
                idxs = np.array([merged_targets[idx] == test_only_class for idx in dict_users_test[user_idx]])
                test_only_class_idxs = dict_users_test[user_idx][idxs] # dataset idx
                if len(test_only_class_idxs) <= k:
                    dict_users_test[user_idx] = dict_users_test[user_idx][~idxs]
                else:
                    random_idxs = np.random.choice(test_only_class_idxs, k, replace=False) # dataset idx
                    mask = np.isin(dict_users_test[user_idx], random_idxs)
                    dict_users_test[user_idx] = dict_users_test[user_idx][~mask]
                    dict_users_train[user_idx] = np.append(dict_users_train[user_idx], random_idxs)
        

        dataset_train = merged_dataset
        dataset_test = merged_dataset
        
        # check if there is any class only in train or test set
        for user in range(total_users):
            user_targets_train = np.array(dataset_train.targets)[dict_users_train[user]]
            user_targets_test = np.array(dataset_test.targets)[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            # check if there is any class that is in test and not in train set
            for c in test_classes:
                if c not in train_classes:
                    logging.info(f"[!] test class {c} is not in train set")
                    raise ValueError(f"[!] test class {c} is not in train set")
                
        logging.info(f"[+] total data size after move data: train {sum([len(dict_users_train[user]) for user in range(total_users)])}, test {sum([len(dict_users_test[user]) for user in range(total_users)])}")

    if copy_data:
        merged_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
        merged_targets = np.concatenate((targets_train, targets_test), axis=0)
        merged_dataset.targets = merged_targets
        dict_users_test_orig = copy.deepcopy(dict_users_test)
        for user_idx in range(total_users):
            dict_users_test[user_idx] = dict_users_test[user_idx] + len(dataset_train)

        # check merged dataset
        for user_idx in range(total_users):
            dict_user_orig = dict_users_test_orig[user_idx]
            dict_user = dict_users_test[user_idx]
            for idx_orig, idx in zip(dict_user_orig, dict_user):
                assert torch.allclose(dataset_test[idx_orig][0], merged_dataset[idx][0])
                assert dataset_test[idx_orig][1] == merged_dataset[idx][1]

        for user_idx in range(total_users):
            user_targets_train = merged_targets[dict_users_train[user_idx]]
            user_targets_test = merged_targets[dict_users_test[user_idx]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            test_only = np.setdiff1d(test_classes, train_classes, assume_unique=True)
            for test_only_class in test_only:
                idxs = np.array([merged_targets[idx] == test_only_class for idx in dict_users_test[user_idx]])
                test_only_class_idxs = dict_users_test[user_idx][idxs] # dataset idx
                random_idxs = np.random.choice(test_only_class_idxs, 1, replace=False) # dataset idx
                dict_users_train[user_idx] = np.append(dict_users_train[user_idx], random_idxs)
        
        dataset_train = merged_dataset
        dataset_test = merged_dataset
        
        # check if there is any class only in train or test set
        for user in range(total_users):
            user_targets_train = np.array(dataset_train.targets)[dict_users_train[user]]
            user_targets_test = np.array(dataset_test.targets)[dict_users_test[user]]
            train_classes = np.unique(user_targets_train)
            test_classes = np.unique(user_targets_test)
            # check if there is any class that is in test and not in train set
            for c in test_classes:
                if c not in train_classes:
                    logging.info(f"t[!] est class {c} is not in train set")
                    raise ValueError(f"[!] test class {c} is not in train set")
                
        logging.info(f"[+] total data size after copy data: train {sum([len(dict_users_train[user]) for user in range(total_users)])}, test {sum([len(dict_users_test[user]) for user in range(total_users)])}")

    return dataset_train, dataset_test, dict_users_train, dict_users_test        



# def get_model(args):
#     if args.model == 'cnn' and args.dataset in ['cifar10', 'cifar100']:
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'mobile' and args.dataset in ['cifar10', 'cifar100']:
#         net_glob = MobileNetCifar(num_classes=args.num_classes).to(args.device)
#     elif args.model == 'resnet18' and args.dataset in ['cifar10', 'cifar100']:
#         net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
#     elif args.model == 'resnet50' and args.dataset in ['cifar10', 'cifar100']:
#         net_glob = ResNet50(num_classes=args.num_classes).to(args.device)
#     elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_glob = CNNMnist(args=args).to(args.device)
#     elif args.model == 'mlp' and args.dataset == 'mnist':
#         net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
#     else:
#         exit('Error: unrecognized model')

#     return net_glob


from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



## MAML++ utils
def get_loss_weights(inner_loop, annealing_epochs, current_epoch):
    """Code from MAML++ paper AntreasAntoniou`s Pytorch Implementation(slightly modified for integration)
    return A tensor to be used to compute the weighted average of the loss, useful for the MSL(multi step loss)
    inner_loop : MAML inner loop number
    annealing_epochs : As learning progresses, weights are increased for higher inner loop
    current_epoch : Current epoch
    """
    loss_weights = np.ones(shape=(inner_loop)) * (1.0 / inner_loop)
    decay_rate = 1.0 / inner_loop / annealing_epochs
    min_value_for_non_final_losses = 0.03 / inner_loop
    for i in range(len(loss_weights) - 1):
        curr_value = np.maximum(loss_weights[i] - (current_epoch * decay_rate), min_value_for_non_final_losses)
        loss_weights[i] = curr_value

    curr_value = np.minimum(
        loss_weights[-1] + (current_epoch * (inner_loop -1) * decay_rate),
        1.0 - ((inner_loop - 1) * min_value_for_non_final_losses))
    loss_weights[-1] = curr_value
    loss_weights = torch.Tensor(loss_weights)
    return loss_weights

## Per-FedAvg utils
def approximate_hessian(w_local, functional, data_batch: Tuple[torch.Tensor, torch.Tensor], grad, delta = 1e-4):
    """Code from Per-FedAvg KarhouTam's Pytorch Implementation(slightly modified for integration)
    return Hessian approximation which preserves all theoretical guarantees of MAML, without requiring access to second-order information(HF-MAML)
    """
    w_local = [torch.Tensor(w.data).detach().clone().requires_grad_(True) for w in w_local]
    criterion = torch.nn.CrossEntropyLoss()
    x, y = data_batch

    wt_1 = [torch.Tensor(w) for w in w_local]
    wt_2 = [torch.Tensor(w) for w in w_local]

    wt_1 = [w + delta*g for w,g in zip(wt_1, grad)]
    wt_2 = [w - delta*g for w,g in zip(wt_1, grad)]

    logit_1 = functional(wt_1, x)
    loss_1 = criterion(logit_1, y)
    grads_1 = torch.autograd.grad(loss_1, w_local)

    logit_2 = functional(wt_2, x)
    loss_2 = criterion(logit_2, y)
    grads_2 = torch.autograd.grad(loss_2, w_local)

    with torch.no_grad():
        grads_2nd = deepcopy(grads_1)
        for g_2nd, g1, g2 in zip(grads_2nd, grads_1, grads_2):
            g_2nd.data = (g1-g2) / (2*delta)

    return grads_2nd

def clip_norm_(grads, max_norm, norm_type: float = 2.0):
    """ This code is based on torch.nn.utils.clip_grad_norm_(inplace function that does gradient clipping to max_norm).
    the input of torch.nn.utils.clip_grad_norm_ is parameters
    but the input of clip_norm_ is list of gradients
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm  + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))

    return total_norm

# def clip_norm_coef(grads, max_norm, norm_type: float = 2.0):
#     max_norm = float(max_norm)
#     norm_type = float(norm_type)

#     device = grads[0].device
#     total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
#     clip_coef = max_norm / (total_norm  + 1e-6)

#     clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

#     return clip_coef_clamped.to(device)

def clip_norm_coef(grads, max_norm, norm_type: float = 2.0):
    """ This code looks similar to torch.nn.utils.clip_grad_norm_ and clip_norm_,
    but it is very different because it does not detach grads(important to MAML algorithm).
    return A scalar coefficient that normalizes the norm of gradients to the max_norm
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for g in grads]), norm_type)
    clip_coef = max_norm / (total_norm  + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)

def clip_norm_coef_wo_logit(grads, max_norm, norm_type: float = 2.0):
    """ This code looks similar to torch.nn.utils.clip_grad_norm_ and clip_norm_,
    but it is very different because it does not detach grads(important to MAML algorithm).
    return A scalar coefficient that normalizes the norm of gradients to the max_norm
    """
    logit_layer_num = 10 # to exclude NVDP logit layer
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    device = grads[0].device
    total_norm = torch.norm(torch.stack([torch.norm(g, norm_type).to(device) for i, g in enumerate(grads) if i!=logit_layer_num]), norm_type)
    clip_coef = max_norm / (total_norm  + 1e-6)

    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)

    return clip_coef_clamped.to(device)



def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds, labels_oneh):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds, labels_oneh)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE


def draw_reliability_graph(preds, labels_oneh):
#   import ipdb; ipdb.set_trace(context=5)
  ECE, MCE = get_metrics(preds, labels_oneh)
  bins, _, bin_accs, _, _ = calc_bins(preds, labels_oneh)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True)
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  #plt.show()
  #plt.savefig('calibrated_network.png', bbox_inches='tight')
  #plt.close(fig)
  return fig, ECE, MCE


def plot_data_partition(dataset, dict_users, num_classes, num_sample_users, writer=None, tag="Data Partition"):
    dict_users_targets={}
    targets=np.array(dataset.targets)

    dict_users_targets = {client_idx: targets[data_idxs] for client_idx, data_idxs in dict_users.items()}

    s=torch.stack([torch.bincount(torch.tensor(data_idxs), minlength=num_classes) for client_idx, data_idxs in dict_users_targets.items()])
    ss=torch.cumsum(s, 1)
    cmap = plt.cm.get_cmap('hsv', num_classes)
    fig, ax = plt.subplots(figsize=(20, num_sample_users))
    ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, 0], color=cmap(0))
    for c in range(1, num_classes):
        ax.barh([f"Client {i:3d}" for i in range(num_sample_users)], s[:num_sample_users, c], left=ss[:num_sample_users, c-1], color=cmap(c))
    # plt.show()
    if writer is not None:
        writer.add_figure(tag, fig)
