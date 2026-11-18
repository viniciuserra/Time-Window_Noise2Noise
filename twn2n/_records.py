"""Internal helpers that normalize the signal containers accepted by the library.

Every public function accepts either one record or a list of records. A record
is an array with one row per time sample and one column per channel. Internally
everything is a list of two-dimensional ``float64`` arrays; the helpers below
convert the input to that form and convert the result back to the form the
caller used.
"""
import numpy as np


def as_records(signals):
    """Convert ``signals`` to a list of 2-D arrays and return a function that restores its form.

    Parameters
    ----------
    signals : array-like or list of array-like
        One record with shape ``(n_samples,)`` or ``(n_samples, n_channels)``,
        a ``pandas.DataFrame`` with one column per channel, or a list (or
        tuple) of such records. Records in a list may have different lengths
        and must have the same number of channels.

    Returns
    -------
    records : list of numpy.ndarray
        Records with shape ``(n_samples, n_channels)`` and ``float64`` dtype.
    restore : callable
        ``restore(list_of_arrays)`` returns the data in the container type and
        shape of ``signals``.
    """
    is_list = isinstance(signals, (list, tuple))
    items = list(signals) if is_list else [signals]
    if is_list and len(items) == 0:
        raise ValueError("The list of records is empty.")

    records, rebuilders = [], []
    for item in items:
        array = np.asarray(item, dtype=float)
        if array.ndim == 1:
            records.append(array.reshape(-1, 1))
            rebuilders.append(lambda out: out[:, 0])
        elif array.ndim == 2:
            records.append(array)
            if hasattr(item, "columns") and hasattr(item, "index"):
                rebuilders.append(_frame_builder(item))
            else:
                rebuilders.append(lambda out: out)
        else:
            raise ValueError(
                "Each record must be one- or two-dimensional, with time along the first axis. "
                "To pass several records, use a list of arrays, one array per record."
            )

    n_channels = {r.shape[1] for r in records}
    if len(n_channels) != 1:
        raise ValueError(f"All records must have the same number of channels; found {sorted(n_channels)}.")
    for r in records:
        if not np.all(np.isfinite(r)):
            raise ValueError("The signals contain NaN or infinite values.")

    def restore(outputs):
        rebuilt = [build(np.asarray(out)) for build, out in zip(rebuilders, outputs)]
        if not is_list:
            return rebuilt[0]
        return tuple(rebuilt) if isinstance(signals, tuple) else rebuilt

    return records, restore


def _frame_builder(frame):
    def build(out):
        if out.shape == frame.shape:
            return type(frame)(out, index=frame.index, columns=frame.columns)
        return out
    return build


def select_channels(record, channels):
    """Columns of ``record`` given by ``channels`` (``None`` keeps every column)."""
    if channels is None:
        return record
    return record[:, list(channels)]
