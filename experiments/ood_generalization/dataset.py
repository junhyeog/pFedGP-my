import copy
import logging
from collections import defaultdict

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

# from train_utils import get_data
from experiments.ood_generalization.train_utils import get_data


def classes_per_node_dirichlet(labels_list, num_users, alpha):
    if isinstance(labels_list, torch.Tensor):
        num_classes = len(labels_list.unique())
    else:
        num_classes = len(np.unique(labels_list))

    # create distribution for each client
    alpha_list = [alpha for _ in range(num_classes)]
    # np.random.seed(42) # ! fixed: do not use seed to generate different partitions on train and test set
    prob_array = np.random.dirichlet(alpha_list, num_users)

    # normalizing
    prob_array /= prob_array.sum(axis=0)

    class_partitions = defaultdict(list)
    cls_list = [i for i in range(num_classes)]
    for i in range(num_users):
        class_partitions["class"].append(cls_list)
        class_partitions["prob"].append(prob_array[i, :])

    return class_partitions


def gen_data_split(labels_list, num_users, class_partitions):
    if isinstance(labels_list, torch.Tensor):
        labels_list = labels_list.cpu().numpy().astype(int)
    num_classes, num_samples = np.unique(labels_list, return_counts=True)

    # -------------------------- #
    # Create class index mapping #
    # -------------------------- #
    data_class_idx = {i: np.where(labels_list == i)[0] for i in range(len(num_classes))}

    # --------- #
    # Shuffling #
    # --------- #
    for data_idx in data_class_idx.values():
        np.random.shuffle(data_idx)

    # ------------------------------ #
    # Assigning samples to each user #
    # ------------------------------ #
    user_data_idx = [[] for _ in range(num_users)]
    for usr_i in range(num_users):
        for c, p in zip(class_partitions["class"][usr_i], class_partitions["prob"][usr_i]):
            end_idx = int(num_samples[c] * p)
            user_data_idx[usr_i].extend(data_class_idx[c][:end_idx])
            data_class_idx[c] = data_class_idx[c][end_idx:]

    return user_data_idx


def copy_test_only_data(dataset_train, dataset_test, train_idxs, val_idxs, test_idxs):
    assert len(train_idxs) == len(val_idxs) == len(test_idxs)
    total_users = len(train_idxs)

    targets_train = np.array(dataset_train.targets)
    targets_test = np.array(dataset_test.targets)

    logging.warning(
        f"[+] total original data size: train {sum([len(train_idxs[user]) for user in range(total_users)])}, val {sum([len(val_idxs[user]) for user in range(total_users)])}, test {sum([len(test_idxs[user]) for user in range(total_users)])}"
    )

    # * merge train and test dataset
    merged_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_test])
    merged_targets = np.concatenate((targets_train, targets_test), axis=0)
    merged_dataset.targets = merged_targets
    test_idx_orig = copy.deepcopy(test_idxs)
    for user_idx in range(total_users):
        test_idxs[user_idx] = test_idxs[user_idx] + len(dataset_train)

    # check merged dataset
    for user_idx in range(total_users):
        for idx_orig, idx in zip(test_idx_orig[user_idx], test_idxs[user_idx]):
            assert torch.allclose(dataset_test[idx_orig][0], merged_dataset[idx][0])
            assert dataset_test[idx_orig][1] == merged_dataset[idx][1]

    dataset_train = merged_dataset
    dataset_test = merged_dataset

    # * test only data
    # move test only data to train set
    for user_idx in range(total_users):
        user_targets_train = merged_targets[train_idxs[user_idx]]
        user_targets_test = merged_targets[test_idxs[user_idx]]
        train_classes = np.unique(user_targets_train)
        test_classes = np.unique(user_targets_test)
        test_only = np.setdiff1d(test_classes, train_classes, assume_unique=True)
        for test_only_class in test_only:
            idxs = np.array([merged_targets[idx] == test_only_class for idx in test_idxs[user_idx]])
            test_only_class_idxs = test_idxs[user_idx][idxs]  # dataset idx
            random_idxs = np.random.choice(test_only_class_idxs, 1, replace=False)  # dataset idx
            train_idxs[user_idx] = np.append(train_idxs[user_idx], random_idxs)

    # check if there is any class only in train or test set
    for user in range(total_users):
        user_targets_train = np.array(dataset_train.targets)[train_idxs[user]]
        user_targets_test = np.array(dataset_test.targets)[test_idxs[user]]
        train_classes = np.unique(user_targets_train)
        test_classes = np.unique(user_targets_test)
        # check if there is any class that is in test and not in train set
        for c in test_classes:
            if c not in train_classes:
                logging.warning(f"t[!] test class {c} is not in train set")
                raise ValueError(f"[!] test class {c} is not in train set")

    # * val only data
    # move val only data to train set
    for user_idx in range(total_users):
        user_targets_train = merged_targets[train_idxs[user_idx]]
        user_targets_test = merged_targets[val_idxs[user_idx]]
        train_classes = np.unique(user_targets_train)
        test_classes = np.unique(user_targets_test)
        test_only = np.setdiff1d(test_classes, train_classes, assume_unique=True)
        for test_only_class in test_only:
            idxs = np.array([merged_targets[idx] == test_only_class for idx in val_idxs[user_idx]])
            test_only_class_idxs = val_idxs[user_idx][idxs]  # dataset idx
            random_idxs = np.random.choice(test_only_class_idxs, 1, replace=False)  # dataset idx
            train_idxs[user_idx] = np.append(train_idxs[user_idx], random_idxs)

    # check if there is any class only in train or val set
    for user in range(total_users):
        user_targets_train = np.array(dataset_train.targets)[train_idxs[user]]
        user_targets_test = np.array(dataset_test.targets)[val_idxs[user]]
        train_classes = np.unique(user_targets_train)
        test_classes = np.unique(user_targets_test)
        # check if there is any class that is in test and not in train set
        for c in test_classes:
            if c not in train_classes:
                logging.warning(f"t[!] val class {c} is not in train set")
                raise ValueError(f"[!] val class {c} is not in train set")

    logging.warning(
        f"[+] total data size after copy data: train {sum([len(train_idxs[user]) for user in range(total_users)])}, val {sum([len(val_idxs[user]) for user in range(total_users)])}, test {sum([len(test_idxs[user]) for user in range(total_users)])}"
    )

    return dataset_train, dataset_test, train_idxs, val_idxs, test_idxs


def create_generalization_loaders(data_name, data_root, num_train_users, num_gen_users, bz, alpha: float = 10, args=None):
    if args.env == "pfedgp":
        logging.warning(f"[+] create_generalization_loaders: env: {args.env} -> use pfedgp dataset")
        # get datasets and idxs of each split
        train_dataset, test_dataset, train_idx, val_idx, test_idx = get_datasets(data_name, data_root)
        # create train / novel nodes partitions
        train_nodes_idx, novel_nodes_idx = idx_partition_per_group(train_idx, val_idx, test_idx, novel_nodes_size=float(num_gen_users / (num_train_users + num_gen_users)))
        # iterate over groups train/novel + different splits train/val/test
        idx_user_split = [[[] for _ in range(3)] for _ in range(2)]
        for g_id, g_nodes_split in enumerate((train_nodes_idx, novel_nodes_idx)):  # iterate over train / novel nodes
            # g_nodes_split holds train/novel train/val/test indexes
            n_users = num_train_users if g_id == 0 else num_gen_users
            for split_id, s in enumerate(g_nodes_split):  # iterate over train/val/test splits
                # assuming train/val/test order
                # check if split is test or not
                if split_id != 2:
                    if isinstance(train_dataset.targets, list):
                        labels_list = np.array(train_dataset.targets)
                    else:
                        labels_list = train_dataset.targets
                    labels_list = labels_list[s]

                else:  # train / val case
                    if isinstance(test_dataset.targets, list):
                        labels_list = np.array(test_dataset.targets)
                    else:
                        labels_list = test_dataset.targets
                    labels_list = labels_list[s]

                if split_id == 0:
                    if g_id == 0:
                        class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha)
                    else:
                        alpha_gen = alpha
                        class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha_gen)
                labels_list_index = gen_data_split(labels_list, n_users, class_partitions)
                idx_user_split[g_id][split_id].extend([np.array(s)[i] for i in labels_list_index])

        # unite groups and create dataloaders
        generalization_loaders = []
        # change order of clientes - first 10 are novel clients
        idx_user_split = idx_user_split[::-1]
        for s_i in range(len(idx_user_split[0])):
            loaders = []
            if s_i != 2:
                data = train_dataset
            else:
                data = test_dataset
            for g_i in range(len(idx_user_split)):
                for u_idx in idx_user_split[g_i][s_i]:
                    loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0), num_workers=4))
            generalization_loaders.append(loaders)
        return generalization_loaders

    elif args.env == "pfedgp1":  # classes_per_node_dirichlet on test set
        logging.warning(f"[+] create_generalization_loaders: env: {args.env} -> use pfedgp1 dataset")
        # get datasets and idxs of each split
        train_dataset, test_dataset, train_idx, val_idx, test_idx = get_datasets(data_name, data_root)
        # create train / novel nodes partitions
        train_nodes_idx, novel_nodes_idx = idx_partition_per_group(train_idx, val_idx, test_idx, novel_nodes_size=float(num_gen_users / (num_train_users + num_gen_users)))
        # iterate over groups train/novel + different splits train/val/test
        idx_user_split = [[[] for _ in range(3)] for _ in range(2)]
        for g_id, g_nodes_split in enumerate((train_nodes_idx, novel_nodes_idx)):  # iterate over train / novel nodes
            # g_nodes_split holds train/novel train/val/test indexes
            n_users = num_train_users if g_id == 0 else num_gen_users
            for split_id, s in enumerate(g_nodes_split):  # iterate over train/val/test splits
                # assuming train/val/test order
                # check if split is test or not
                if split_id != 2:
                    if isinstance(train_dataset.targets, list):
                        labels_list = np.array(train_dataset.targets)
                    else:
                        labels_list = train_dataset.targets
                    labels_list = labels_list[s]

                else:  # train / val case
                    if isinstance(test_dataset.targets, list):
                        labels_list = np.array(test_dataset.targets)
                    else:
                        labels_list = test_dataset.targets
                    labels_list = labels_list[s]
                # >>> original code
                # if split_id == 0:
                # <<<
                # ! >>> fixed: classes_per_node_dirichlet on test set
                if True:
                    # ! <<<
                    if g_id == 0:
                        class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha)
                    else:
                        alpha_gen = alpha
                        class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha_gen)
                labels_list_index = gen_data_split(labels_list, n_users, class_partitions)
                idx_user_split[g_id][split_id].extend([np.array(s)[i] for i in labels_list_index])

        # copy test only data
        # 1. concat train / novel users
        # 2. copy test only data to train set
        # 3. split train / novel users
        concat_train_idxs = np.concatenate((idx_user_split[0][0], idx_user_split[1][0]), axis=0)
        concat_val_idxs = np.concatenate((idx_user_split[0][1], idx_user_split[1][1]), axis=0)
        concat_test_idxs = np.concatenate((idx_user_split[0][2], idx_user_split[1][2]), axis=0)
        train_dataset, test_dataset, train_idxs, val_idxs, test_idxs = copy_test_only_data(train_dataset, test_dataset, concat_train_idxs, concat_val_idxs, concat_test_idxs)
        idx_user_split = [
            [train_idxs[:num_train_users], val_idxs[:num_train_users], test_idxs[:num_train_users]],
            [train_idxs[num_train_users:], val_idxs[num_train_users:], test_idxs[num_train_users:]],
        ]

        # unite groups and create dataloaders
        generalization_loaders = []
        # change order of clientes - first 10 are novel clients
        idx_user_split = idx_user_split[::-1]
        for s_i in range(len(idx_user_split[0])):
            loaders = []
            if s_i != 2:
                data = train_dataset
            else:
                data = test_dataset
            for g_i in range(len(idx_user_split)):
                for u_idx in idx_user_split[g_i][s_i]:
                    loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0), num_workers=4))
            generalization_loaders.append(loaders)
        return generalization_loaders

    elif args.env == "pfedgp2":  # do not split ood data
        logging.warning(f"[+] create_generalization_loaders: env: {args.env} -> use pfedgp2 dataset")
        # get datasets and idxs of each split
        train_dataset, test_dataset, train_idx, val_idx, test_idx = get_datasets(data_name, data_root)
        nodes_idx = (train_idx, val_idx, test_idx)
        # iterate over groups train/novel + different splits train/val/test
        idx_user_split = [[] for _ in range(3)]
        # g_nodes_split holds train/novel train/val/test indexes
        n_users = num_train_users + num_gen_users
        for split_id, s in enumerate(nodes_idx):  # iterate over train/val/test splits
            # assuming train/val/test order
            # check if split is test or not
            if split_id != 2:
                if isinstance(train_dataset.targets, list):
                    labels_list = np.array(train_dataset.targets)
                else:
                    labels_list = train_dataset.targets
                labels_list = labels_list[s]

            else:  # train / val case
                if isinstance(test_dataset.targets, list):
                    labels_list = np.array(test_dataset.targets)
                else:
                    labels_list = test_dataset.targets
                labels_list = labels_list[s]

            if split_id == 0:
                class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha)
            labels_list_index = gen_data_split(labels_list, n_users, class_partitions)
            idx_user_split[split_id].extend([np.array(s)[i] for i in labels_list_index])

        train_idx, val_idx, test_idx = idx_user_split
        idx_user_split = [
            [train_idx[:num_train_users], val_idx[:num_train_users], test_idx[:num_train_users]],
            [train_idx[num_train_users:], val_idx[num_train_users:], test_idx[num_train_users:]],
        ]

        # unite groups and create dataloaders
        generalization_loaders = []
        # change order of clientes - first 10 are novel clients
        idx_user_split = idx_user_split[::-1]
        for s_i in range(len(idx_user_split[0])):
            loaders = []
            if s_i != 2:
                data = train_dataset
            else:
                data = test_dataset
            for g_i in range(len(idx_user_split)):
                for u_idx in idx_user_split[g_i][s_i]:
                    loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0), num_workers=4))
            generalization_loaders.append(loaders)
        return generalization_loaders

    elif args.env == "pfedgp3":  # do not split ood data & classes_per_node_dirichlet on test set
        logging.warning(f"[+] create_generalization_loaders: env: {args.env} -> use pfedgp3 dataset")
        # get datasets and idxs of each split
        train_dataset, test_dataset, train_idx, val_idx, test_idx = get_datasets(data_name, data_root)
        nodes_idx = (train_idx, val_idx, test_idx)
        # iterate over groups train/novel + different splits train/val/test
        idx_user_split = [[] for _ in range(3)]
        # g_nodes_split holds train/novel train/val/test indexes
        n_users = num_train_users + num_gen_users
        for split_id, s in enumerate(nodes_idx):  # iterate over train/val/test splits
            # assuming train/val/test order
            # check if split is test or not
            if split_id != 2:
                if isinstance(train_dataset.targets, list):
                    labels_list = np.array(train_dataset.targets)
                else:
                    labels_list = train_dataset.targets
                labels_list = labels_list[s]

            else:  # train / val case
                if isinstance(test_dataset.targets, list):
                    labels_list = np.array(test_dataset.targets)
                else:
                    labels_list = test_dataset.targets
                labels_list = labels_list[s]

            class_partitions = classes_per_node_dirichlet(labels_list, num_users=n_users, alpha=alpha)
            labels_list_index = gen_data_split(labels_list, n_users, class_partitions)
            idx_user_split[split_id].extend([np.array(s)[i] for i in labels_list_index])

        # copy test only data
        # 1. copy test only data to train set
        # 2. split train / novel users
        train_idx, val_idx, test_idx = idx_user_split
        train_dataset, test_dataset, train_idxs, val_idxs, test_idxs = copy_test_only_data(train_dataset, test_dataset, train_idx, val_idx, test_idx)
        idx_user_split = [
            [train_idxs[:num_train_users], val_idxs[:num_train_users], test_idxs[:num_train_users]],
            [train_idxs[num_train_users:], val_idxs[num_train_users:], test_idxs[num_train_users:]],
        ]

        # unite groups and create dataloaders
        generalization_loaders = []
        # change order of clientes - first 10 are novel clients
        idx_user_split = idx_user_split[::-1]
        for s_i in range(len(idx_user_split[0])):
            loaders = []
            if s_i != 2:
                data = train_dataset
            else:
                data = test_dataset
            for g_i in range(len(idx_user_split)):
                for u_idx in idx_user_split[g_i][s_i]:
                    loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0), num_workers=4))
            generalization_loaders.append(loaders)
        return generalization_loaders

    elif "bmfl" in args.env:
        logging.warning(f"[+] create_generalization_loaders: env: {args.env} -> use bmfl dataset")
        idx_user_split = [[[] for _ in range(3)] for _ in range(2)]
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(data_name, num_train_users, num_gen_users, alpha, args=args)
        train_dataset = dataset_train
        test_dataset = dataset_test

        # first group: participating clients
        for i in range(num_train_users):
            idx_user_split[0][0].append(dict_users_train[i])
            idx_user_split[0][1].append(dict_users_test[i])  # ! use test for validation
            idx_user_split[0][2].append(dict_users_test[i])

        # second group: non-participating clients
        for i in range(num_gen_users):
            idx_user_split[1][0].append(dict_users_train[num_train_users + i])
            idx_user_split[1][1].append(dict_users_test[num_train_users + i])  # ! use test for validation
            idx_user_split[1][2].append(dict_users_test[num_train_users + i])

        # unite groups and create dataloaders
        generalization_loaders = []
        # change order of clientes - first 10 are novel clients # ! checked
        idx_user_split = idx_user_split[::-1]
        for s_i in range(len(idx_user_split[0])):
            loaders = []
            if s_i == 0:  # train
                data = train_dataset
            else:  # val/test
                data = test_dataset
            for g_i in range(len(idx_user_split)):
                for u_idx in idx_user_split[g_i][s_i]:
                    loaders.append(DataLoader(Subset(data, u_idx), bz, (s_i == 0), num_workers=4))
            generalization_loaders.append(loaders)
        return generalization_loaders


def idx_partition_per_group(train_idx, val_idx, test_idx, novel_nodes_size=0.2):
    train_nodes_idx, novel_nodes_idx = [], []
    for id in (train_idx, val_idx, test_idx):
        t_nodes_idx, n_nodes_idx = train_test_split(range(len(id)), test_size=novel_nodes_size, random_state=42)
        train_nodes_idx.append(t_nodes_idx)
        novel_nodes_idx.append(n_nodes_idx)
    return train_nodes_idx, novel_nodes_idx


def get_datasets(data_name, dataroot, normalize=True, val_size=10000):
    if data_name == "cifar10":
        normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        data_obj = CIFAR10
    elif data_name == "cifar100":
        normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
        data_obj = CIFAR100

    else:
        raise ValueError(f'data_name should be one of {["cifar10", "cifar100"]}')

    trans = [transforms.ToTensor()]

    if normalize:
        trans.append(normalization)

    transform = transforms.Compose(trans)

    train_set = data_obj(dataroot, train=True, download=True, transform=transform)

    test_set = data_obj(dataroot, train=False, download=True, transform=transform)

    train_size = len(train_set) - val_size
    train_idx, val_idx = train_test_split(range(len(train_set)), train_size=train_size, random_state=42)
    test_idx = list(range(len(test_set)))

    return train_set, test_set, train_idx, val_idx, test_idx
