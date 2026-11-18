"""The TWN2N denoiser: build the masked windows, train the autoencoder, return the denoised signals."""
import json
import os
import random
import time
import warnings

import numpy as np

from ._records import as_records
from .network import build_network, stop_after_overfitting
from .windows import _check_p, split_rows, stack_windows


class TWN2N:
    """Time-Window Noise2Noise: self-supervised blind denoising of multichannel signals.

    The network is trained on the noisy signals themselves. Each sample is
    predicted from the ``p`` samples before it and the ``p`` samples after it
    in every channel, while the sample itself is hidden from the input, and the
    prediction passes through a bottleneck of ``r`` units. No clean
    reference and no description of the noise are needed.

    The first and the last ``p`` samples of each record have no complete
    window. They are not used as training targets and are returned as measured.

    Parameters
    ----------
    r : int
        Number of units of the bottleneck (the latent dimension of the
        paper). It should be smaller than the number of channels, so that the
        code has to compress the state of the system. The paper uses 4 for
        six channels and 3 for five channels.
    p : int, default 2
        Number of neighbors on each side of the masked instant. The input of
        the network has ``2 * p * n_channels`` features.
    network : {"standard", "compact"} or callable, default "standard"
        Architecture. ``"standard"`` is the one of the vibration benchmark and
        ``"compact"`` the one of the impact benchmark. A callable
        ``f(n_inputs, n_outputs, r)`` that returns an uncompiled Keras
        model is also accepted.
    epochs : int, default 5000
        Maximum number of epochs. The stopping rule normally ends the training
        much earlier (a few hundred epochs in the paper).
    batch_size : int, default 32
    learning_rate : float, default 1e-3
        Learning rate of the Adam optimizer.
    patience : int, default 100
        Patience of the stopping rule (see
        :func:`twn2n.network.stop_after_overfitting`).
    validation_fraction : float, default 0.1
        Fraction of the windows held out, at random, to monitor the stopping
        rule.
    split_seed : int, default 42
        Seed of the random split between training and validation windows.
    seed : int or None, default None
        Seed of the weight initialization and of the batch shuffling. When
        given, the global generators of Python, NumPy, and TensorFlow are
        seeded with it.
    verbose : int, default 0
        Verbosity passed to Keras: 0 is silent, 1 shows a progress bar, and 2
        prints one line per epoch.

    Attributes
    ----------
    model_ : keras.Model
        The trained network.
    r_ : int
        Bottleneck size that was used.
    n_channels_ : int
    n_parameters_ : int
        Number of trainable parameters.
    epochs_run_ : int
        Number of epochs that were run.
    best_epoch_ : int
        Epoch (counted from 1) with the lowest validation loss, whose weights
        are kept.
    training_time_s_ : float
    history_ : dict
        Training and validation loss per epoch (keys ``"loss"`` and
        ``"val_loss"``), in standardized units.

    Examples
    --------
    >>> import twn2n
    >>> model = twn2n.TWN2N(p=2, r=4, seed=0)
    >>> denoised = model.fit_denoise(noisy)      # noisy: (n_samples, 6) array
    """

    def __init__(self, r, p=2, network="standard", epochs=5000, batch_size=32,
                 learning_rate=1e-3, patience=100, validation_fraction=0.1, split_seed=42,
                 seed=None, verbose=0):
        self.p = _check_p(p)
        if isinstance(r, bool) or int(r) != r or r < 1:
            raise ValueError(f"r must be a positive integer; got {r!r}.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie between 0 and 1.")
        self.r = int(r)
        self.network = network
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.validation_fraction = float(validation_fraction)
        self.split_seed = int(split_seed)
        self.seed = None if seed is None else int(seed)
        self.verbose = verbose
        self.model_ = None

    # ------------------------------------------------------------------ fit --
    def fit(self, signals):
        """Train the network on the noisy signals.

        Parameters
        ----------
        signals : array-like or list of array-like
            One record with shape ``(n_samples, n_channels)`` (or
            ``(n_samples,)`` for a single channel), or a list of records of
            the same system. Records in a list may have different lengths.

        Returns
        -------
        self
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from tensorflow import keras

        records, _ = as_records(signals)
        X, Y, _ = stack_windows(records, self.p)
        n_channels = Y.shape[1]
        r = self.r
        if n_channels > 1 and r >= n_channels:
            warnings.warn(
                f"r = {r} is not smaller than the number of channels ({n_channels}). "
                "The masked input still prevents the identity map, but the bottleneck no longer "
                "compresses the output.", stacklevel=2)

        if self.seed is not None:
            _seed_everything(int(self.seed))

        scaler_x, scaler_y = StandardScaler(), StandardScaler()
        Xs, Ys = scaler_x.fit_transform(X), scaler_y.fit_transform(Y)
        X_train, X_val, Y_train, Y_val = train_test_split(
            Xs, Ys, test_size=self.validation_fraction, random_state=self.split_seed)

        model = build_network(X.shape[1], n_channels, r, self.network)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                      loss="mean_absolute_error")
        callback = stop_after_overfitting(self.patience)
        start = time.time()
        batches = _shuffled_batches(X_train, Y_train, self.batch_size, self.seed)
        history = model.fit(batches, validation_data=(X_val, Y_val), epochs=self.epochs, shuffle=False,
                            callbacks=[callback], verbose=self.verbose)     # `batches` shuffles by itself
        self.training_time_s_ = time.time() - start

        self.model_ = model
        self.r_ = int(r)
        self.n_channels_ = int(n_channels)
        self.n_parameters_ = int(model.count_params())
        self.epochs_run_ = len(history.history["loss"])
        self.best_epoch_ = int(callback.best_epoch + 1)
        self.best_val_loss_ = float(callback.best_loss)
        self.history_ = {"loss": [float(v) for v in history.history["loss"]],
                         "val_loss": [float(v) for v in history.history["val_loss"]]}
        self._scaler_x = (scaler_x.mean_.copy(), scaler_x.scale_.copy())
        self._scaler_y = (scaler_y.mean_.copy(), scaler_y.scale_.copy())
        return self

    # -------------------------------------------------------------- denoise --
    def denoise(self, signals):
        """Denoise signals with the trained network.

        ``signals`` are usually the records given to :meth:`fit`. Other records
        of the same system, with the same channels, sampling rate, and units,
        can also be passed.

        The first and the last ``p`` samples of each record have no complete
        window, so they are returned as measured.

        Returns
        -------
        denoised : same container and shape as ``signals``
        """
        self._check_fitted()
        records, restore = as_records(signals)
        X, Y, lengths = stack_windows(records, self.p)
        if Y.shape[1] != self.n_channels_:
            raise ValueError(f"The network was trained with {self.n_channels_} channels; got {Y.shape[1]}.")
        Xs = (X - self._scaler_x[0]) / self._scaler_x[1]
        prediction = np.asarray(self.model_.predict(Xs, verbose=0), dtype=float)
        rows = prediction * self._scaler_y[1] + self._scaler_y[0]
        denoised = []
        for record, interior in zip(records, split_rows(rows, lengths)):
            output = record.copy()
            output[self.p:len(record) - self.p] = interior
            denoised.append(output)
        return restore(denoised)

    def fit_denoise(self, signals):
        """Train on ``signals`` and return their denoised version."""
        return self.fit(signals).denoise(signals)

    # ----------------------------------------------------------- persistence --
    def save(self, path):
        """Save the trained denoiser to a single ``.npz`` file.

        The file stores the settings, the standardization constants, and the
        weights as plain NumPy arrays, so it does not depend on the Keras file
        format. Custom architectures passed as a callable are not stored; pass
        the same callable to :meth:`load`.
        """
        self._check_fitted()
        settings = dict(p=self.p, r=self.r_, epochs=self.epochs,
                        network=self.network if isinstance(self.network, str) else None,
                        batch_size=self.batch_size, learning_rate=self.learning_rate,
                        patience=self.patience, validation_fraction=self.validation_fraction,
                        split_seed=self.split_seed, seed=self.seed, n_channels=self.n_channels_,
                        epochs_run=self.epochs_run_, best_epoch=self.best_epoch_,
                        best_val_loss=self.best_val_loss_, training_time_s=self.training_time_s_)
        arrays = {f"weight_{i:02d}": w for i, w in enumerate(self.model_.get_weights())}
        np.savez_compressed(path, settings=json.dumps(settings), x_mean=self._scaler_x[0],
                            x_scale=self._scaler_x[1], y_mean=self._scaler_y[0],
                            y_scale=self._scaler_y[1], loss=np.asarray(self.history_["loss"]),
                            val_loss=np.asarray(self.history_["val_loss"]), **arrays)

    @classmethod
    def load(cls, path, network=None):
        """Load a denoiser saved with :meth:`save`."""
        path = os.fspath(path)
        if not os.path.exists(path) and os.path.exists(path + ".npz"):
            path = path + ".npz"                # numpy adds the extension when save() receives a bare name
        with np.load(path, allow_pickle=False) as data:
            settings = json.loads(str(data["settings"]))
            net = network if network is not None else settings["network"]
            if net is None:
                raise ValueError("This file was saved with a custom architecture; pass it as `network`.")
            obj = cls(r=settings["r"], p=settings["p"], network=net,
                      epochs=settings["epochs"], batch_size=settings["batch_size"],
                      learning_rate=settings["learning_rate"], patience=settings["patience"],
                      validation_fraction=settings["validation_fraction"],
                      split_seed=settings["split_seed"], seed=settings["seed"])
            obj.n_channels_ = int(settings["n_channels"])
            obj.r_ = int(settings["r"])
            obj._scaler_x = (data["x_mean"], data["x_scale"])
            obj._scaler_y = (data["y_mean"], data["y_scale"])
            obj.history_ = {"loss": data["loss"].tolist(), "val_loss": data["val_loss"].tolist()}
            weights = [data[k] for k in sorted(k for k in data.files if k.startswith("weight_"))]
        obj.epochs_run_ = int(settings["epochs_run"])
        obj.best_epoch_ = int(settings["best_epoch"])
        obj.best_val_loss_ = float(settings["best_val_loss"])
        obj.training_time_s_ = float(settings["training_time_s"])
        model = build_network(2 * obj.p * obj.n_channels_, obj.n_channels_, obj.r_, net)
        model.set_weights(weights)
        obj.model_ = model
        obj.n_parameters_ = int(model.count_params())
        return obj

    # --------------------------------------------------------------- helpers --
    def _check_fitted(self):
        if self.model_ is None:
            raise RuntimeError("This TWN2N instance is not trained yet; call fit() or fit_denoise() first.")

    def __repr__(self):
        net = self.network if isinstance(self.network, str) else "custom"
        return f"TWN2N(p={self.p}, r={self.r}, network={net!r})"


def denoise(signals, r, p=2, return_model=False, **kwargs):
    """Denoise signals in one call.

    Shortcut for ``TWN2N(r=r, p=p, **kwargs).fit_denoise(signals)``.

    Parameters
    ----------
    signals : array-like or list of array-like
        Noisy record or list of records (see :meth:`TWN2N.fit`).
    r : int
        Number of units of the bottleneck.
    p : int, default 2
        Number of neighbors on each side of the masked instant.
    return_model : bool, default False
        If ``True``, also return the trained :class:`TWN2N` object.
    **kwargs
        Any other argument of :class:`TWN2N`.

    Returns
    -------
    denoised : same container and shape as ``signals``
    model : TWN2N
        Only when ``return_model`` is ``True``.
    """
    model = TWN2N(r=r, p=p, **kwargs)
    denoised = model.fit_denoise(signals)
    return (denoised, model) if return_model else denoised


def _shuffled_batches(X, Y, batch_size, seed=None):
    """Training batches, drawn in a new random order at every epoch.

    The paper trains with the windows reshuffled at every epoch. Keras 2 does
    that inside ``model.fit``. Keras 3 repeats the order of the first epoch in
    all the following ones whenever the global TensorFlow seed is set, which
    changes the training and lowers the SNR of the output. Shuffling here, with
    ``reshuffle_each_iteration=True``, gives the behavior of the paper with both
    versions and keeps the training reproducible when ``seed`` is given.
    """
    import tensorflow as tf
    from tensorflow import keras

    dtype = keras.backend.floatx()
    x, y = tf.constant(X, dtype=dtype), tf.constant(Y, dtype=dtype)
    n = int(x.shape[0])
    order = tf.data.Dataset.range(n).shuffle(n, seed=seed, reshuffle_each_iteration=True)
    batches = order.batch(int(batch_size)).map(lambda index: (tf.gather(x, index), tf.gather(y, index)))
    return batches.prefetch(1)


def _seed_everything(seed):
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed % (2 ** 31))
    tf.random.set_seed(seed)
