"""Autoencoder architectures and the stopping rule used to train them.

TensorFlow is imported only when a network is built, so the rest of the
library (noise, metrics, filters, benchmarks) works without it.
"""
import numpy as np

NETWORKS = ("standard", "compact")


def layer_sizes(n_inputs, n_outputs, r, network="standard"):
    """Number of units of every dense layer, from the first hidden layer to the output.

    Parameters
    ----------
    n_inputs : int
        Number of input features, ``2 * p * n_channels``.
    n_outputs : int
        Number of output features, ``n_channels``.
    r : int
        Number of units of the bottleneck (the latent dimension of the paper).
    network : {"standard", "compact"}
        ``"standard"`` is the architecture of the vibration benchmark of the
        paper (24-102-54-54-12-[4]-12-30-54-54-6 for six channels, ``p = 2``,
        and ``r = 4``). ``"compact"`` is the smaller architecture of
        the impact benchmark (20-45-25-25-3-[3]-3-25-25-45-5 for five
        channels, ``p = 2``, and ``r = 3``).

    Returns
    -------
    sizes : list of int
        Units per layer. The bottleneck is the fifth entry.
    """
    n, r = int(n_inputs), int(r)
    if network == "standard":
        neck = max(1, min(int(n * 0.5), 2 * r + 6))
        return [4 * n + 6, 2 * n + 6, 2 * n + 6, neck, r, neck, n + 6, 2 * n + 6, 2 * n + 6, int(n_outputs)]
    if network == "compact":
        neck = max(1, min(int(n * 0.5), r))
        return [2 * n + 5, n + 5, n + 5, neck, r, neck, n + 5, n + 5, 2 * n + 5, int(n_outputs)]
    raise ValueError(f"Unknown network {network!r}; choose one of {NETWORKS} or pass a callable.")


def build_network(n_inputs, n_outputs, r, network="standard"):
    """Build the Keras model of the TWN2N autoencoder.

    All hidden layers use the hyperbolic tangent; the output layer is linear.
    ``network`` may also be a callable ``f(n_inputs, n_outputs, r)``
    that returns an uncompiled Keras model, for custom architectures.
    """
    if callable(network):
        return network(n_inputs, n_outputs, r)
    from tensorflow import keras

    sizes = layer_sizes(n_inputs, n_outputs, r, network)
    inputs = keras.layers.Input(shape=(int(n_inputs),))
    x = inputs
    for units in sizes[:-1]:
        x = keras.layers.Dense(int(units), activation="tanh")(x)
    outputs = keras.layers.Dense(int(sizes[-1]), activation="linear")(x)
    return keras.Model(inputs, outputs, name=f"twn2n_{network}_{int(r)}d")


def stop_after_overfitting(patience=100):
    """Create the stopping callback of the paper.

    The callback keeps the weights of the epoch with the lowest validation
    loss. Training stops when two conditions hold at the same time: the
    validation loss has not improved for ``patience`` epochs, and it is larger
    than the training loss. The best weights are then restored. Because the
    target of TWN2N is the noisy sample itself, driving the loss to its minimum
    would reproduce the noise; this rule stops the training before that.

    If the maximum number of epochs is reached first, the best weights are
    restored as well. With the settings of the paper (5,000 epochs) the rule
    always stops the training earlier.
    """
    from tensorflow import keras

    class StopAfterOverfitting(keras.callbacks.Callback):
        def __init__(self, patience):
            super().__init__()
            self.patience = int(patience)
            self.wait = 0
            self.stopped_epoch = 0
            self.best_weights = None
            self.best_loss = np.inf
            self.best_epoch = 0

        def on_train_begin(self, logs=None):
            self.wait = 0
            self.stopped_epoch = 0
            self.best_loss = np.inf
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_loss = logs.get("val_loss")
            train_loss = logs.get("loss")
            if val_loss is None:
                return
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
            if self.wait >= self.patience and val_loss > train_loss:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.best_weights is not None:
                    self.model.set_weights(self.best_weights)

        def on_train_end(self, logs=None):
            if self.stopped_epoch == 0 and self.best_weights is not None:
                self.model.set_weights(self.best_weights)

    return StopAfterOverfitting(patience)
