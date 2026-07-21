"""Time-Window Noise2Noise (TWN2N): blind-spot dataset, autoencoder, training, inference."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


def lagged_dataframe(signals, num_points=2):
    """Build the time-window (lag/lead) feature table for one experiment.

    For every channel `x` it creates x_m_p ... x_m_1, x, x_p_1 ... x_p_p.
    Boundary samples are padded with the first/last value.
    """
    df = pd.DataFrame(signals)
    out = pd.DataFrame()
    for col in df.columns:
        s = df[col]
        first, last = s.iloc[0], s.iloc[-1]
        for i in range(num_points, 0, -1):
            out[f"{col}_m_{i}"] = s.shift(i, fill_value=first)
        out[col] = s
        for i in range(1, num_points + 1):
            out[f"{col}_p_{i}"] = s.shift(-i, fill_value=last)
    return out


def tensor_to_dataframe(S, num_points=2):
    """Stack all excitations of an (n_samples, n_channels, n_inputs) tensor."""
    frames = []
    for i in range(S.shape[2]):
        sig = {f"x{j + 1}": S[:, j, i] for j in range(S.shape[1])}
        frames.append(lagged_dataframe(sig, num_points))
    return pd.concat(frames, ignore_index=True)


def blind_spot_columns(n_channels=6, num_points=2):
    """Input columns exclude the central instant (the blind spot); targets are the centre."""
    inputs = []
    for c in range(1, n_channels + 1):
        for i in range(num_points, 0, -1):
            inputs.append(f"x{c}_m_{i}")
        for i in range(1, num_points + 1):
            inputs.append(f"x{c}_p_{i}")
    targets = [f"x{c}" for c in range(1, n_channels + 1)]
    return inputs, targets


def build_model(n_in, n_out, latent_dim=4):
    """Compact autoencoder with a bottleneck strictly smaller than the output."""
    inp = Input(shape=(n_in,))
    x = Dense(int(n_in * 4 + 6), activation="tanh")(inp)
    x = Dense(int(n_in * 2 + 6), activation="tanh")(x)
    x = Dense(int(n_in * 2 + 6), activation="tanh")(x)
    x = Dense(int(min(n_in * 0.5, 2 * latent_dim + 6)), activation="tanh")(x)
    x = Dense(latent_dim, activation="tanh")(x)                      # latent code r
    x = Dense(int(min(n_in * 0.5, 2 * latent_dim + 6)), activation="tanh")(x)
    x = Dense(int(n_in * 1 + 6), activation="tanh")(x)
    x = Dense(int(n_in * 2 + 6), activation="tanh")(x)
    x = Dense(int(n_in * 2 + 6), activation="tanh")(x)
    out = Dense(n_out, activation="linear")(x)
    return Model(inp, out, name=f"TWN2N_{latent_dim}D")


def denoise(S_noisy, num_points=2, latent_dim=4, epochs=1000, batch_size=256,
            patience=60, learning_rate=1e-3, verbose=0):
    """Blind-denoise a noisy tensor (n_samples, n_channels, n_inputs).

    Trained only on the noisy signal itself: the target is the noisy central sample,
    which the network never sees at its input.
    """
    n_samples, n_channels, n_inputs = S_noisy.shape
    df = tensor_to_dataframe(S_noisy, num_points)
    in_cols, out_cols = blind_spot_columns(n_channels, num_points)

    X, y = df[in_cols].values, df[out_cols].values
    sx, sy = StandardScaler(), StandardScaler()
    Xs, ys = sx.fit_transform(X), sy.fit_transform(y)

    X_tr, X_val, y_tr, y_val = train_test_split(Xs, ys, test_size=0.1, random_state=42)
    model = build_model(len(in_cols), len(out_cols), latent_dim)
    model.compile(Adam(learning_rate=learning_rate), loss="mean_absolute_error")
    stopper = EarlyStopping(monitor="val_loss", patience=patience,
                            restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs,
              batch_size=batch_size, callbacks=[stopper], verbose=verbose)

    pred = sy.inverse_transform(model.predict(Xs, verbose=0))
    S_den = pred.reshape((n_inputs, n_samples, n_channels)).transpose(1, 2, 0)
    return S_den, model
