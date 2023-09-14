import argparse
import json
import logging
import os
import random
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed, cudnn_enabled=True):
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

    torch.backends.cudnn.enabled = cudnn_enabled
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logger():
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.WARNING)


def get_device(cuda=True, gpus="0"):
    return torch.device("cuda:" + gpus if torch.cuda.is_available() and cuda else "cpu")


def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def take(X, Y, classes):
    indices = np.isin(Y, classes)
    return X[indices], Y[indices]


def pytorch_take(X, Y, classes):
    indices = torch.stack([y_ == Y for y_ in classes]).sum(0).bool()
    return X[indices], Y[indices]


def lbls1_to_lbls2(Y, l2l):
    for lbls1_class, lbls2_class in l2l.items():
        if isinstance(lbls2_class, list):
            for c in lbls2_class:
                Y[Y == lbls1_class] = c + 1000
        elif isinstance(lbls2_class, int):
            Y[Y == lbls1_class] = lbls2_class + 1000
        else:
            raise NotImplementedError("not a valid type")

    return Y - 1000


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


# create folders for saving models and logs
def _init_(out_path, exp_name):
    script_path = os.path.dirname(__file__)
    script_path = "." if script_path == "" else script_path
    if not os.path.exists(out_path + "/" + exp_name):
        os.makedirs(out_path + "/" + exp_name)
    # save configurations
    os.system("cp -r " + script_path + "/*.py " + out_path + "/" + exp_name)


def get_art_dir(args):
    art_dir = Path(args.out_dir)
    art_dir.mkdir(exist_ok=True, parents=True)

    curr = 0
    existing = [int(x.as_posix().split("_")[-1]) for x in art_dir.iterdir() if x.is_dir()]
    if len(existing) > 0:
        curr = max(existing) + 1

    out_dir = art_dir / f"version_{curr}"
    out_dir.mkdir()

    return out_dir


def save_experiment(args, results, return_out_dir=False, save_results=True):
    out_dir = get_art_dir(args)

    json.dump(vars(args), open(out_dir / "meta.experiment", "w"))

    # loss curve
    if save_results:
        json.dump(results, open(out_dir / "results.experiment", "w"))

    if return_out_dir:
        return out_dir


def topk(true, pred, k):
    max_pred = np.argsort(pred, axis=1)[:, -k:]  # take top k
    two_d_true = np.expand_dims(true, 1)  # 1d -> 2d
    two_d_true = np.repeat(two_d_true, k, axis=1)  # repeat along second axis
    return (two_d_true == max_pred).sum() / true.shape[0]


def to_one_hot(y, dtype=torch.double):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], y.max().type(torch.IntTensor) + 1), dtype=dtype, device=y.device)
    return y_output_onehot.scatter_(1, y.unsqueeze(1), 1)


def CE_loss(y, y_hat, num_classes, reduction="mean"):
    # convert a single label into a one-hot vector
    y_output_onehot = torch.zeros((y.shape[0], num_classes), dtype=y_hat.dtype, device=y.device)
    y_output_onehot.scatter_(1, y.unsqueeze(1), 1)
    if reduction == "mean":
        return -torch.sum(y_output_onehot * torch.log(y_hat + 1e-12), dim=1).mean()
    return -torch.sum(y_output_onehot * torch.log(y_hat + 1e-12))


def permute_data_lbls(data, labels):
    perm = np.random.permutation(data.shape[0])
    return data[perm], labels[perm]


def N_vec(y):
    """
    Compute the count vector for PG Multinomial inference
    :param x:
    :return:
    """
    if y.dim() == 1:
        N = torch.sum(y)
        reminder = N - torch.cumsum(y)[:-2]
        return torch.cat((torch.tensor([N]).to(y.device), reminder))
    elif y.dim() == 2:
        N = torch.sum(y, dim=1, keepdim=True)
        reminder = N - torch.cumsum(y, dim=1)[:, :-2]
        return torch.cat((N, reminder), dim=1)
    else:
        raise ValueError("x must be 1 or 2D")


def kappa_vec(y):
    """
    Compute the kappa vector for PG Multinomial inference
    :param x:
    :return:
    """
    if y.dim() == 1:
        return y[:-1] - N_vec(y) / 2.0
    elif y.dim() == 2:
        return y[:, :-1] - N_vec(y) / 2.0
    else:
        raise ValueError("x must be 1 or 2D")


# modified from:
# https://github.com/cornellius-gp/gpytorch/blob/master/gpytorch/utils/cholesky.py
def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
    """Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
    Args:
        :attr:`A` (Tensor):
            The tensor to compute the Cholesky decomposition of
        :attr:`upper` (bool, optional):
            See torch.cholesky
        :attr:`out` (Tensor, optional):
            See torch.cholesky
        :attr:`jitter` (float, optional):
            The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
            as 1e-6 (float) or 1e-8 (double)
    """
    try:
        L = torch.cholesky(A, upper=upper, out=out)
        return L
    except RuntimeError as e:
        isnan = torch.isnan(A)
        if isnan.any():
            raise ValueError(f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN.")

        if jitter is None:
            jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
        Aprime = A.clone()
        jitter_prev = 0
        for i in range(5):
            jitter_new = jitter * (10**i)
            Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
            jitter_prev = jitter_new
            try:
                L = torch.cholesky(Aprime, upper=upper, out=out)
                warnings.warn(
                    f"A not p.d., added jitter of {jitter_new} to the diagonal",
                    RuntimeWarning,
                )
                return L
            except RuntimeError:
                continue
        raise e


def print_calibration(ECE_module, out_dir, lbls_vs_target, file_name, color, temp=1.0):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(), color=color, temp=temp)
    logging.info(f"{file_name}, " f"ECE: {ece_metrics[0].item():.3f}, " f"MCE: {ece_metrics[1].item():.3f}, " f"BRI: {ece_metrics[2].item():.3f}")


def calibration_search(ECE_module, out_dir, lbls_vs_target, color, file_name):
    lbls_preds = torch.tensor(lbls_vs_target)
    probs = lbls_preds[:, 1:]
    targets = lbls_preds[:, 0]

    temps = torch.tensor([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    eces = [ECE_module.forward(probs, targets, None, color=color, temp=t)[0].item() for t in temps]
    best_temp = round(temps[np.argmin(eces)].item(), 2)

    ece_metrics = ECE_module.forward(probs, targets, (out_dir / file_name).as_posix(), color=color, temp=best_temp)
    logging.info(
        f"{file_name}, " f"Best Temperature: {best_temp:.3f}, " f"ECE: {ece_metrics[0].item():.3f}, " f"MCE: {ece_metrics[1].item():.3f}, " f"BRI: {ece_metrics[2].item():.3f}"
    )

    return best_temp


def offset_client_classes(loader, device):
    for i, batch in enumerate(loader):
        img, label = tuple(t.to(device) for t in batch)
        all_labels = label if i == 0 else torch.cat((all_labels, label))

    client_labels, client_indices = torch.sort(torch.unique(all_labels))
    label_map = {client_labels[i].item(): client_indices[i].item() for i in range(client_labels.shape[0])}
    return label_map


def calc_metrics(results):
    if len(results) == 0:
        return 0.0, 0.0

    total_correct = sum([val["correct"] for val in results.values()])
    total_samples = sum([val["total"] for val in results.values()])
    avg_loss = np.mean([val["loss"] for val in results.values()])
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def calc_weighted_metrics(results, client_data_size):
    if len(results) == 0:
        return 0.0, 0.0

    user_idxs = list(results.keys())
    weights_size = []
    for user_idx in user_idxs:
        weights_size.append(client_data_size[user_idx])
    weights = np.array(weights_size) / sum(weights_size)

    avg_loss = np.average([results[user_idx]["loss"] for user_idx in user_idxs], weights=weights)
    avg_acc = np.average([results[user_idx]["correct"] / results[user_idx]["total"] for user_idx in user_idxs], weights=weights)

    return avg_loss, avg_acc


def calc_bins(preds, labels_oneh):
  # Assign each prediction to a bin
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins) # array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])
  binned = np.digitize(preds, bins) # (-inf, 0.1) -> 0, [0.1, 0.2) -> 1, ..., [1.0, inf) -> 10

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin] # y-axis: relative frequncy
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin] # x-axis : mean pred value of bin 

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
