"""Tests of the library. Run with ``pytest`` from the root of the repository.

The fast tests cover the windows, the noise models, the metrics, the filters,
the benchmarks, and the modal identification. The tests marked ``slow`` train
small networks; skip them with ``pytest -m "not slow"``.
"""
import numpy as np
import pytest

import twn2n
from twn2n import benchmarks, filters, modal
from twn2n.network import layer_sizes
from twn2n.windows import split_rows, stack_windows


@pytest.fixture(scope="module")
def vibration():
    system = benchmarks.three_dof_system()
    t, clean = benchmarks.three_dof_impulse_responses(system)
    return system, t, clean


# ------------------------------------------------------------------ windows --
def test_blind_spot_windows_exclude_the_central_sample():
    record = np.arange(20, dtype=float).reshape(10, 2)          # channel 0: 0, 2, 4...; channel 1: 1, 3, 5...
    X, Y = twn2n.blind_spot_windows(record, p=2)
    # the first and the last p samples have no complete window and are not targets
    assert X.shape == (6, 8) and Y.shape == (6, 2)
    # first window: target at t = 2 (values 4 and 5), neighbors t-2, t-1, t+1, t+2 of each channel
    assert Y[0].tolist() == [4, 5]
    assert X[0].tolist() == [0, 2, 6, 8, 1, 3, 7, 9]
    assert Y[-1].tolist() == [14, 15]
    assert X[-1].tolist() == [10, 12, 16, 18, 11, 13, 17, 19]


@pytest.mark.parametrize("p", [1, 2, 3])
def test_no_target_appears_in_its_own_input(p):
    record = np.arange(60, dtype=float).reshape(20, 3)          # every value is unique
    X, Y = twn2n.blind_spot_windows(record, p=p)
    assert len(Y) == 20 - 2 * p
    for row, target in zip(X, Y):
        assert not set(row.tolist()) & set(target.tolist())


def test_windows_do_not_cross_records():
    a, b = np.ones((6, 1)), 5 * np.ones((8, 1))
    X, Y, lengths = stack_windows([a, b], p=1)
    assert lengths == [4, 6]
    assert set(np.unique(X[:4])) == {1.0} and set(np.unique(X[4:])) == {5.0}
    parts = split_rows(Y, lengths)
    assert parts[0].shape == (4, 1) and parts[1].shape == (6, 1)


def test_invalid_window_arguments():
    with pytest.raises(ValueError):
        twn2n.blind_spot_windows(np.zeros((4, 1)), p=2)
    with pytest.raises(ValueError):
        twn2n.blind_spot_windows(np.zeros((40, 1)), p=0)


# ------------------------------------------------------------------ network --
def test_layer_sizes_match_the_paper():
    assert layer_sizes(24, 6, 4, "standard") == [102, 54, 54, 12, 4, 12, 30, 54, 54, 6]
    assert layer_sizes(20, 5, 3, "compact") == [45, 25, 25, 3, 3, 3, 25, 25, 45, 5]


def test_default_r_matches_the_paper():
    assert twn2n.default_r(6) == 4
    assert twn2n.default_r(5) == 3
    assert twn2n.default_r(1) == 1


# -------------------------------------------------------------------- noise --
@pytest.mark.parametrize("kind", ["white", "pink", "brown"])
def test_additive_noise_has_the_requested_snr(vibration, kind):
    _, _, clean = vibration
    noisy = twn2n.add_noise(clean, kind, snr_db=12.0, rng=1)
    per_record = [twn2n.snr_db(c, n, per_channel=True) for c, n in zip(clean, noisy)]
    assert np.allclose(per_record, 12.0, atol=1e-9)
    assert isinstance(noisy, list) and noisy[0].shape == clean[0].shape


def test_noise_is_reproducible_and_selective(vibration):
    _, _, clean = vibration
    a = twn2n.add_noise(clean[0], "white", 20.0, rng=3, channels=[0, 1])
    b = twn2n.add_noise(clean[0], "white", 20.0, rng=3, channels=[0, 1])
    assert np.array_equal(a, b)
    assert np.array_equal(a[:, 2:], clean[0][:, 2:]) and not np.array_equal(a[:, :2], clean[0][:, :2])


def test_noise_spectra_have_the_expected_slopes():
    f = np.fft.rfftfreq(2 ** 14)[1:]
    slopes = {}
    for kind in ("white", "pink", "brown"):
        psd = np.mean([np.abs(np.fft.rfft(twn2n.colored_noise(kind, 2 ** 14, rng=s))[1:]) ** 2 for s in range(40)], axis=0)
        band = (f > 1e-3) & (f < 1e-1)
        slopes[kind] = np.polyfit(np.log10(f[band]), np.log10(psd[band]), 1)[0]
    assert abs(slopes["white"]) < 0.1
    assert abs(slopes["pink"] + 1.0) < 0.1
    assert abs(slopes["brown"] + 2.0) < 0.15


def test_quantization_of_the_vibration_benchmark(vibration):
    _, _, clean = vibration
    quantized = twn2n.add_noise(clean, "quantization")
    assert abs(twn2n.snr_db(clean, quantized, channels=[0, 1, 2]) - 15.36) < 0.01     # value reported in the paper
    x = quantized[0][:, 0]
    assert len(np.unique(np.round(x / np.min(np.abs(x[x != 0])), 6))) <= 16           # the peak spans about seven levels


def test_unknown_noise_kind():
    with pytest.raises(ValueError):
        twn2n.add_noise(np.zeros((10, 1)), "violet")


# ------------------------------------------------------------------ metrics --
def test_snr_definition():
    rng = np.random.default_rng(0)
    clean = rng.standard_normal((5000, 2))
    noise = rng.standard_normal((5000, 2))
    noise = noise / noise.std() * clean.std() / 10.0
    assert abs(twn2n.snr_db(clean, clean + noise) - 20.0) < 1e-9
    assert abs(twn2n.snr_gain_db(clean, clean + noise, clean + noise / 2) - 20 * np.log10(2)) < 1e-9
    assert twn2n.snr_db(clean, clean + noise, per_channel=True).shape == (2,)


def test_band_snr_separates_bands():
    fs, n = 1000.0, 4000
    t = np.arange(n) / fs
    clean = np.sin(2 * np.pi * 50 * t)
    estimate = clean + 0.1 * np.sin(2 * np.pi * 300 * t)       # the error lives at 300 Hz only
    assert twn2n.band_snr_db(clean, estimate, (0, 100), fs) > 60
    assert twn2n.band_snr_db(clean, estimate, (0, 400), fs) == pytest.approx(20.0, abs=0.1)


# ------------------------------------------------------------------ filters --
def test_filters_improve_white_noise(vibration):
    system, _, clean = vibration
    noisy = twn2n.add_noise(clean, "white", 15.0, rng=0)
    low = filters.lowpass(noisy, 1.05 * system.fn[2], system.fs)
    sg = filters.savitzky_golay(noisy, 5, 3)
    assert twn2n.snr_gain_db(clean, noisy, low, channels=[0, 1, 2]) > 3.0
    assert twn2n.snr_gain_db(clean, noisy, sg, channels=[0, 1, 2]) > 2.5
    assert low[0].shape == noisy[0].shape


def test_visushrink_runs(vibration):
    pytest.importorskip("pywt")
    _, _, clean = vibration
    noisy = twn2n.add_noise(clean[0], "white", 5.0, rng=0)
    out = filters.visushrink(noisy, channels=[0, 1, 2])
    assert out.shape == noisy.shape
    assert np.array_equal(out[:, 3:], noisy[:, 3:])
    assert twn2n.snr_db(clean[0], out, channels=[0, 1, 2]) > twn2n.snr_db(clean[0], noisy, channels=[0, 1, 2])


# --------------------------------------------------------------- benchmarks --
def test_three_dof_system_matches_the_paper(vibration):
    system, t, clean = vibration
    assert np.allclose(system.fn, [40.15, 120.10, 239.91], atol=0.005)
    assert np.allclose(system.zeta, [0.0100, 0.0072, 0.0100], atol=5e-5)
    assert system.duration == 3.0 and system.fs == 1024.0
    assert len(clean) == 3 and clean[0].shape == (3072, 6) and len(t) == 3072


def test_modal_identification_on_clean_signals(vibration):
    system, _, clean = vibration
    modes = modal.identify_modes(clean[0][:, :3], system.fs)
    matches = modal.match_modes(modes, system.wn, system.zeta)
    assert all(m["found"] for m in matches)
    assert max(m["wn_error_pct"] for m in matches) < 0.01
    assert max(m["zeta_error_pct"] for m in matches) < 1.0


def test_impact_benchmark_is_consistent():
    records, times = benchmarks.hunt_crossley_impacts(velocities=(0.3,), restitutions=(0.9,), n_eval=100_000)
    record = records[0]
    assert record.shape[1] == 5 and len(times[0]) == len(record)
    x, v, f_kn, e, v0 = record.T
    assert np.all(x >= -1e-9) and f_kn.max() > 0
    assert np.allclose(benchmarks.hunt_crossley_force(x, v, e, v0) / 1000.0, f_kn, rtol=1e-9, atol=1e-12)
    assert np.all(e == 0.9) and np.all(v0 == 0.3)


# ----------------------------------------------------------------- denoiser --
def _mixture_of_two_modes(n=1500, seed=0):
    """Four channels that are mixtures of two damped oscillations (two independent components)."""
    t = np.arange(n) / 500.0
    modes = np.column_stack([np.exp(-0.8 * t) * np.sin(2 * np.pi * 9 * t),
                             np.exp(-0.5 * t) * np.sin(2 * np.pi * 4 * t)])
    clean = modes @ np.array([[1.0, 0.5], [1.0, -0.7], [0.4, 1.0], [-0.8, 0.6]]).T
    return clean, twn2n.add_noise(clean, "white", 10.0, rng=seed)


@pytest.mark.slow
def test_denoiser_improves_the_snr_and_keeps_the_container():
    clean, noisy = _mixture_of_two_modes()
    model = twn2n.TWN2N(p=2, r=2, epochs=60, seed=0)
    denoised = model.fit_denoise(noisy)
    assert denoised.shape == noisy.shape
    assert twn2n.snr_gain_db(clean, noisy, denoised) > 3.0
    assert model.n_parameters_ > 0 and 1 <= model.best_epoch_ <= model.epochs_run_ <= 60
    assert len(model.history_["loss"]) == model.epochs_run_

    # the first and the last p samples of a record are returned as measured
    assert np.array_equal(denoised[:2], noisy[:2]) and np.array_equal(denoised[-2:], noisy[-2:])
    assert not np.any(denoised[2:-2] == noisy[2:-2])

    as_list = model.denoise([noisy[:700], noisy[700:]])
    assert isinstance(as_list, list) and as_list[0].shape == (700, 4) and as_list[1].shape == (800, 4)
    assert np.array_equal(as_list[1][:2], noisy[700:702])
    single = twn2n.TWN2N(p=1, epochs=2, seed=0).fit_denoise(noisy[:, 0])
    assert single.shape == (len(noisy),)
    shortcut, fitted = twn2n.denoise(noisy, p=1, r=2, epochs=2, seed=0, return_model=True)
    assert shortcut.shape == noisy.shape and fitted.p == 1


@pytest.mark.slow
def test_batches_are_reshuffled_at_every_epoch():
    """Keras 3 repeats the first order when a global seed is set; the library must not."""
    from tensorflow import keras

    from twn2n.denoiser import _seed_everything, _shuffled_batches

    def epoch_orders():
        orders = []

        class Recorder(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                orders.append([])

        class Probe(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(1)

            def call(self, inputs):
                return self.dense(inputs)

            def train_step(self, data):
                orders[-1].extend(np.asarray(data[0])[:, 0].astype(int).tolist())
                return super().train_step(data)

        _seed_everything(3)
        x = np.arange(12, dtype=float).reshape(-1, 1)
        model = Probe()
        model.compile(optimizer="sgd", loss="mse", run_eagerly=True)
        model.fit(_shuffled_batches(x, x, 4, seed=3), epochs=4, verbose=0, callbacks=[Recorder()])
        return orders

    first, second = epoch_orders(), epoch_orders()
    assert all(sorted(order) == list(range(12)) for order in first)        # every sample once per epoch
    assert len({tuple(order) for order in first}) == 4                     # a new order at every epoch
    assert first == second                                                 # reproducible with a seed


@pytest.mark.slow
def test_save_and_load(tmp_path):
    _, noisy = _mixture_of_two_modes(n=600)
    model = twn2n.TWN2N(p=2, r=2, network="compact", epochs=3, seed=1).fit(noisy)
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = twn2n.TWN2N.load(path)
    assert np.allclose(loaded.denoise(noisy), model.denoise(noisy), atol=1e-6)
    assert loaded.p == 2 and loaded.r_ == 2 and loaded.n_parameters_ == model.n_parameters_


@pytest.mark.slow
def test_standard_network_has_the_parameter_count_of_the_paper(vibration):
    _, _, clean = vibration
    noisy = twn2n.add_noise(clean, "white", 15.0, rng=0)
    model = twn2n.TWN2N(p=2, r=4, epochs=1, seed=0).fit(noisy)
    assert model.n_parameters_ == 17218


def test_dataframe_round_trip(vibration):
    pd = pytest.importorskip("pandas")
    _, _, clean = vibration
    frame = pd.DataFrame(clean[0], columns=list(benchmarks.THREE_DOF_CHANNELS))
    noisy = twn2n.add_noise(frame, "white", 15.0, rng=0)
    assert isinstance(noisy, pd.DataFrame) and list(noisy.columns) == list(frame.columns)
    assert np.allclose(twn2n.snr_db(frame, noisy, per_channel=True), 15.0, atol=1e-9)


@pytest.mark.slow
def test_command_line(tmp_path):
    from twn2n.__main__ import main, read_table

    _, noisy = _mixture_of_two_modes(n=400)
    time = np.arange(len(noisy)) / 500.0
    source, target = tmp_path / "noisy.csv", tmp_path / "denoised.csv"
    np.savetxt(source, np.column_stack([time, noisy]), delimiter=",", header="t,a,b,c,d", comments="")
    assert main([str(source), str(target), "--columns", "1,2,3,4", "--r", "2", "--epochs", "2", "--seed", "0"]) == 0
    header, data = read_table(target)
    assert header == "t,a,b,c,d" and data.shape == (400, 5)
    assert np.allclose(data[:, 0], time) and not np.allclose(data[:, 1:], noisy)


def test_consolidated_results_give_the_tables_of_the_paper():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "paper" / "tables.py"
    spec = importlib.util.spec_from_file_location("paper_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables = {title.split(".")[0]: rows for title, (_, rows) in module.build_tables().items()}
    assert tables["Table 1"][2] == ["15", "21.01", "0.68", "6.01", "6.02", "4.21", "7.12"]
    assert tables["Table 2"][2] == ["15", "21.01", "6.01", "18.25", "3.25", "5.67", "1.06"]
    assert tables["Table 4"][1][:5] == ["Pink", "15.00", "17.48", "0.66", "2.48"]
    assert tables["Table 7"][5][-1] == "0.0099 ± 0.0011"
    assert tables["Table 11"][0][:5] == ["30.46", "0.024", "0.013", "44 to 49", "20/20"]


def test_errors_before_training_and_on_bad_input():
    with pytest.raises(RuntimeError):
        twn2n.TWN2N().denoise(np.zeros((50, 2)))
    with pytest.raises(ValueError):
        twn2n.TWN2N(p=0)
    with pytest.raises(ValueError):
        twn2n.snr_db(np.zeros((10, 2, 3)), np.zeros((10, 2, 3)))
