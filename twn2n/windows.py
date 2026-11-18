"""Blind-spot time windows: the input features and the targets of TWN2N."""
import numpy as np


def blind_spot_windows(record, p=2):
    """Build the masked time windows of one record.

    For every instant ``t_i`` that has ``p`` samples on each side, the input
    row gathers, channel by channel, the ``p`` samples before and the ``p``
    samples after ``t_i``. The sample at ``t_i`` itself is left out (the blind
    spot) and becomes the target.

    Parameters
    ----------
    record : array-like, shape (n_samples, n_channels)
        One record, with time along the first axis.
    p : int
        Number of neighbors taken on each side of the masked instant.

    Returns
    -------
    X : numpy.ndarray, shape (n_samples - 2 * p, 2 * p * n_channels)
        Input features. For each channel the columns are ordered as
        ``[t_{i-p}, ..., t_{i-1}, t_{i+1}, ..., t_{i+p}]``.
    Y : numpy.ndarray, shape (n_samples - 2 * p, n_channels)
        Targets: the samples at the masked instants ``t_p, ..., t_{n-p-1}``.

    Notes
    -----
    The first and the last ``p`` instants of a record have no complete window
    and are not used as targets. Filling the missing neighbors with copies of
    the end samples would place the target itself in its own input, which the
    blind spot is meant to prevent.
    """
    record = np.asarray(record, dtype=float)
    if record.ndim == 1:
        record = record.reshape(-1, 1)
    if record.ndim != 2:
        raise ValueError("A record must have shape (n_samples, n_channels).")
    p = _check_p(p)
    n_samples, n_channels = record.shape
    if n_samples <= 2 * p:
        raise ValueError(f"A record needs more than 2 * p = {2 * p} samples; it has {n_samples}.")

    stop = n_samples - p
    columns = []
    for channel in range(n_channels):
        s = record[:, channel]
        for lag in range(p, 0, -1):
            columns.append(s[p - lag:stop - lag])
        for lead in range(1, p + 1):
            columns.append(s[p + lead:stop + lead])
    return np.stack(columns, axis=1), record[p:stop].copy()


def stack_windows(records, p=2):
    """Windows of several records, stacked along the rows.

    Windows never cross the boundary between two records.

    Returns
    -------
    X, Y : numpy.ndarray
        Stacked inputs and targets (see :func:`blind_spot_windows`).
    lengths : list of int
        Number of windows of each record (its number of samples minus
        ``2 * p``), used to split the rows again.
    """
    pairs = [blind_spot_windows(r, p) for r in records]
    X = np.vstack([x for x, _ in pairs])
    Y = np.vstack([y for _, y in pairs])
    return X, Y, [len(y) for _, y in pairs]


def split_rows(rows, lengths):
    """Inverse of the stacking done by :func:`stack_windows`."""
    edges = np.cumsum([0] + list(lengths))
    return [rows[a:b] for a, b in zip(edges[:-1], edges[1:])]


def _check_p(p):
    if int(p) != p or p < 1:
        raise ValueError(f"p must be a positive integer; got {p!r}.")
    return int(p)
