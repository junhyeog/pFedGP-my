import logging

from experiments.ood_generalization.dataset import create_generalization_loaders
from utils import set_seed


class GenBaseClients:
    def __init__(self, data_name, data_path, n_clients, n_gen_clients, batch_size=128, alpha=1, args=None, **kwargs):
        self.data_name = data_name
        self.data_path = data_path
        self.n_clients = n_clients
        self.n_gen_nodes = n_gen_clients
        self.alpha = alpha
        self.args = args

        self.batch_size = batch_size

        self.train_loaders, self.val_loaders, self.test_loaders = None, None, None
        n_train_users = self.n_clients - self.n_gen_nodes

        set_seed(self.args.seed)
        logging.warning(f"[+] GenBaseClients(env={self.args.env}): set_seed({self.args.seed})")
        self.train_loaders, self.val_loaders, self.test_loaders = create_generalization_loaders(
            self.data_name,
            self.data_path,
            n_train_users,
            self.n_gen_nodes,
            self.batch_size,
            self.alpha,
            args=self.args,
        )

        # check total, mean, std data size
        self.total_train_data = sum([len(loader.dataset) for loader in self.train_loaders])
        self.total_val_data = sum([len(loader.dataset) for loader in self.val_loaders])
        self.total_test_data = sum([len(loader.dataset) for loader in self.test_loaders])
        self.mean_train_data = self.total_train_data / len(self.train_loaders)
        self.mean_val_data = self.total_val_data / len(self.val_loaders)
        self.mean_test_data = self.total_test_data / len(self.test_loaders)
        self.std_train_data = sum([(len(loader.dataset) - self.mean_train_data) ** 2 for loader in self.train_loaders])
        self.std_val_data = sum([(len(loader.dataset) - self.mean_val_data) ** 2 for loader in self.val_loaders])
        self.std_test_data = sum([(len(loader.dataset) - self.mean_test_data) ** 2 for loader in self.test_loaders])
        self.std_train_data = (self.std_train_data / len(self.train_loaders)) ** 0.5
        self.std_val_data = (self.std_val_data / len(self.val_loaders)) ** 0.5
        self.std_test_data = (self.std_test_data / len(self.test_loaders)) ** 0.5
        logging.warning(f"[+] GenBaseClients(alpha={self.alpha}): (train): total={self.total_train_data:}, mean={self.mean_train_data}, std={self.std_train_data}")
        logging.warning(f"[+] GenBaseClients(alpha={self.alpha}): (val)  : total={self.total_val_data}, mean={self.mean_val_data}, std={self.std_val_data}")
        logging.warning(f"[+] GenBaseClients(alpha={self.alpha}): (test) : total={self.total_test_data}, mean={self.mean_test_data}, std={self.std_test_data}")

    def __len__(self):
        return self.n_clients
