import numpy as np
from scipy.spatial.distance import cdist


def get_lids(X, k=10, batch=None):
    if batch is None:
        lid_batch = mle_batch(X, None, k=k)
    else:
        lid_batch = mle_batch(X, batch, k=k)

    lids = np.asarray(lid_batch, dtype=np.float32)
    return lids


# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    if batch is None:
        k = min(k, len(data) - 1)
    else:
        k = min(k, len(batch))
    f = lambda v: - k / np.sum(np.log(v / v[-1]))

    if batch is None:
        a = cdist(data, data)
        # get the closest k neighbours
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    else:
        batch = np.asarray(batch, dtype=np.float32)
        a = cdist(data, batch)
        # get the closest k neighbours
        a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 0:k]

    a = np.apply_along_axis(f, axis=1, arr=a)
    return a


def get_lids_with_neighbors(X, k=10):
    lid_batch, sort_indices = mle_batch_neighbors(X, X, k=k)
    lids = np.asarray(lid_batch, dtype=np.float32)
    neighbor_lids = lids[sort_indices]
    avg_neighbor_lids = np.mean(neighbor_lids, axis=1)
    lid_ratio = np.divide(avg_neighbor_lids, lids)
    return lids, lid_ratio


# lid of a batch of query points X as well its neighbors
def mle_batch_neighbors(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=a)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    a = a[tuple(idx)]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a, sort_indices
