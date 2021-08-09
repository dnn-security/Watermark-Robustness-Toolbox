import numpy as np
from torch.utils import data
from tqdm import tqdm


def collect_n_samples(n: int,
                      data_loader: data.DataLoader,
                      class_label: int = None,
                      has_labels: bool = True,
                      reduce_labels: bool = False,
                      verbose=True):
    """ Collects n samples from a data loader.
    :param n Number of samples to load. Set to 'np.inf' for all samples.
    :param data_loader The data loader to load the examples from
    :param class_label Load only examples from this target class
    :param has_labels Does the dataset have labels?
    :param reduce_labels Reduce labels.
    :param verbose Show the progress bar
    """
    x_samples, y_samples = [], []
    with tqdm(desc=f"Collecting samples: 0/{n}", total=n, disable=not verbose) as pbar:
        if has_labels:
            for (x, y) in data_loader:
                if len(x_samples) >= n:
                    break
                # Reduce soft labels.
                y_full = y.clone()
                if y.dim() > 1:
                    y = y.argmax(dim=1)

                # Compute indices of samples we want to keep.
                idx = np.arange(x.shape[0])
                if class_label:
                    idx, = np.where(y == class_label)

                if len(idx) > 0:
                    x_samples.extend(x[idx].detach().cpu().numpy())
                    if reduce_labels:
                        y_samples.extend(y[idx].detach().cpu().numpy())
                    else:
                        y_samples.extend(y_full[idx].detach().cpu().numpy())
                    pbar.n = len(x_samples)
                    pbar.refresh()
                    pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")

            if n == np.inf:
                return np.asarray(x_samples), np.asarray(y_samples)

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n]), np.asarray(y_samples[:n])
        else:   # No labels.
            for x in data_loader:
                x_samples.extend(x.detach().cpu().numpy())
                pbar.set_description(f"Collecting samples: {min(len(x_samples)+1, n)}/{n}")
                pbar.update(len(x_samples))
                if len(x_samples) >= n:
                    break

            if len(x_samples) < n:
                print(f"[WARNING]: Could not find enough samples. (Found: {len(x_samples)}, Expected: {n})")
            return np.asarray(x_samples[:n])

