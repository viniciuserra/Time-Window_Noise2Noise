"""The two numerical benchmarks of the paper.

* A linear mass-spring-damper chain with three degrees of freedom (3-DOF),
  excited by an impulse at each mass.
* A set of twenty impacts governed by the nonlinear contact law of Hunt and
  Crossley.

Both return clean signals. Use :func:`twn2n.add_noise` to corrupt them.
"""
from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.integrate import solve_ivp

# ------------------------------------------------------------ 3-DOF chain --
THREE_DOF_CHANNELS = ("x1", "x2", "x3", "a1", "a2", "a3")
THREE_DOF_MASSES = (3.0, 3.0, 3.0)                                   # kg
THREE_DOF_STIFFNESSES = (302000.0, 1067000.0, 2998000.0, 284000.0)   # N/m
THREE_DOF_FS = 1024.0                                                # Hz

#: Damping configurations of the paper. A pair gives the target damping ratios
#: of the first and third modes (Rayleigh damping); ("alpha", z) is
#: mass-proportional damping with ratio z at the first mode and ("beta", z) is
#: stiffness-proportional damping with ratio z at the third mode.
THREE_DOF_DAMPING = {
    "R010": (0.010, 0.010),      # main configuration
    "R002": (0.002, 0.002),
    "R005": (0.005, 0.005),
    "R020": (0.020, 0.020),
    "R050": (0.050, 0.050),
    "M020": ("alpha", 0.020),
    "K020": ("beta", 0.020),
}


@dataclass
class ThreeDOFSystem:
    """Matrices and modal parameters of the 3-DOF chain."""

    damping: str
    M: np.ndarray
    C: np.ndarray
    K: np.ndarray
    alpha: float
    beta: float
    wn: np.ndarray        # natural frequencies, rad/s
    fn: np.ndarray        # natural frequencies, Hz
    zeta: np.ndarray      # damping ratios
    fs: float             # sampling frequency, Hz
    duration: float       # record length, s


def chain_matrices(masses, stiffnesses):
    """Mass and stiffness matrices of a chain fixed at both ends (springs k1 to k4)."""
    m1, m2, m3 = masses
    k1, k2, k3, k4 = stiffnesses
    M = np.diag([m1, m2, m3]).astype(float)
    K = np.array([[k1 + k2, -k2, 0.0],
                  [-k2, k2 + k3, -k3],
                  [0.0, -k3, k3 + k4]], dtype=float)
    return M, K


def state_matrices(M, C, K):
    """State-space matrices ``A`` and ``B`` of ``M x'' + C x' + K x = f``."""
    n = M.shape[0]
    M_inv = np.linalg.inv(M)
    A = np.vstack([np.hstack([np.zeros((n, n)), np.eye(n)]),
                   np.hstack([-M_inv @ K, -M_inv @ C])])
    B = np.vstack([np.zeros((n, n)), M_inv])
    return A, B


def modal_parameters(M, C, K):
    """Natural frequencies (rad/s), damping ratios, and mode shapes from the state matrix."""
    A, _ = state_matrices(M, C, K)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    n = M.shape[0]
    index = np.where(eigenvalues.imag > 0)[0]
    index = index[np.argsort(np.abs(eigenvalues[index]))]
    wn = np.abs(eigenvalues[index])
    zeta = -eigenvalues[index].real / wn
    return wn, zeta, eigenvectors[:n, index]


def three_dof_system(damping="R010"):
    """The 3-DOF chain of the paper.

    Three masses of 3 kg, natural frequencies of 40.15, 120.10, and 239.91 Hz,
    and Rayleigh damping ``C = alpha * M + beta * K``. In the main
    configuration (``"R010"``) the damping ratios are 0.0100, 0.0072, and
    0.0100. The record lasts seven time constants of the first mode (3 s in
    the main configuration) and is sampled at 1024 Hz.

    Parameters
    ----------
    damping : str, default "R010"
        Key of :data:`THREE_DOF_DAMPING`.
    """
    M, K = chain_matrices(THREE_DOF_MASSES, THREE_DOF_STIFFNESSES)
    wn_undamped = np.sqrt(np.sort(np.linalg.eigvalsh(np.linalg.inv(M) @ K)))
    alpha, beta = _rayleigh_coefficients(wn_undamped, THREE_DOF_DAMPING[damping])
    C = alpha * M + beta * K
    wn, zeta, _ = modal_parameters(M, C, K)
    time_constant = 1.0 / (zeta[0] * wn[0])
    duration = float(np.clip(np.ceil(7 * time_constant * 2) / 2, 1.0, 8.0))
    return ThreeDOFSystem(damping=damping, M=M, C=C, K=K, alpha=alpha, beta=beta, wn=wn,
                          fn=wn / (2 * np.pi), zeta=zeta, fs=THREE_DOF_FS, duration=duration)


def three_dof_impulse_responses(system=None):
    """Clean impulse responses of the 3-DOF chain.

    A unit impulse is applied to each mass in turn, which gives three records.

    Parameters
    ----------
    system : ThreeDOFSystem or None
        ``None`` uses the main configuration, :func:`three_dof_system`.

    Returns
    -------
    t : numpy.ndarray, shape (n_samples,)
        Time, in s.
    records : list of numpy.ndarray
        Three records with shape ``(n_samples, 6)``. The columns are the
        displacements ``x1, x2, x3`` (m) and the accelerations ``a1, a2, a3``
        (m/s^2) of the three masses.
    """
    system = system or three_dof_system()
    n = system.M.shape[0]
    A, B = state_matrices(system.M, system.C, system.K)
    t = np.arange(0, system.duration, 1 / system.fs)
    records = []
    for i in range(n):
        model = signal.StateSpace(A, B[:, i:i + 1], np.eye(2 * n), np.zeros((2 * n, 1)))
        _, state = signal.impulse(model, T=t)
        records.append(np.concatenate((state[:, :n], state @ A[n:, :].T), axis=1))
    return t, records


def _rayleigh_coefficients(wn, spec):
    w1, w3 = wn[0], wn[-1]
    if spec[0] == "alpha":
        return 2 * spec[1] * w1, 0.0
    if spec[0] == "beta":
        return 0.0, 2 * spec[1] / w3
    z1, z3 = spec
    alpha, beta = np.linalg.solve(np.array([[1 / (2 * w1), w1 / 2], [1 / (2 * w3), w3 / 2]]),
                                  np.array([z1, z3]))
    return float(alpha), float(beta)


# ------------------------------------------------ Hunt and Crossley impact --
IMPACT_CHANNELS = ("x", "v", "F_kN", "e", "v_approach")
IMPACT_MASS = 0.54                                        # kg
IMPACT_STIFFNESS = 2.4614e8                               # N/m^1.5
IMPACT_VELOCITIES = (0.1, 0.15, 0.3, 0.4, 0.5)            # m/s
IMPACT_RESTITUTIONS = tuple(np.linspace(0.76, 0.97, 4))


def hunt_crossley_force(x, v, e, v_approach, stiffness=IMPACT_STIFFNESS):
    """Contact force of the Hunt and Crossley law, in N.

    ``F = K x^1.5 (1 + chi / K * v)`` with ``chi = 3 K (1 - e) / (2 v_approach)``
    for positive indentation ``x``, and zero otherwise.
    """
    x, v = np.asarray(x, dtype=float), np.asarray(v, dtype=float)
    chi = 3 * stiffness * (1 - np.asarray(e, dtype=float)) / (2 * np.asarray(v_approach, dtype=float))
    force = stiffness * np.power(np.clip(x, 0, None), 1.5) * (1 + chi / stiffness * v)
    return np.where(x <= 0, 0.0, force)


def hunt_crossley_impacts(velocities=IMPACT_VELOCITIES, restitutions=IMPACT_RESTITUTIONS,
                          n_eval=1_000_000):
    """Clean signals of the impact benchmark.

    A mass of 0.54 kg hits a flat surface with each combination of approach
    velocity and coefficient of restitution (twenty impacts by default). Each
    record covers the contact, from the first touch to the separation.

    Parameters
    ----------
    velocities : sequence of float
        Approach velocities, in m/s.
    restitutions : sequence of float
        Nominal coefficients of restitution of the contact law.
    n_eval : int, default 1,000,000
        Number of output instants in the 0.1 s integration span. The contact
        ends long before, so each record keeps only the first few thousand.
        The default gives the 299,522 samples of the paper.

    Returns
    -------
    records : list of numpy.ndarray
        One record per impact, with shape ``(n_samples, 5)``. The columns are
        the indentation ``x`` (m), the indentation velocity ``v`` (m/s), the
        contact force ``F_kN`` (kN), and two columns that are constant inside
        a record and identify the test: the coefficient of restitution ``e``
        and the approach velocity ``v_approach`` (m/s).
    times : list of numpy.ndarray
        Time vector of each record, in s.
    """
    records, times = [], []
    for v0 in velocities:
        for e in restitutions:
            chi = 3 * IMPACT_STIFFNESS * (1 - e) / (2 * v0)

            def rhs(t, state, chi=chi):
                x, v = state
                force = 0.0 if x <= 0 else IMPACT_STIFFNESS * x ** 1.5 + chi * x ** 1.5 * v
                return [v, -force / IMPACT_MASS]

            def separation(t, state):
                return state[0]
            separation.terminal, separation.direction = True, -1

            solution = solve_ivp(rhs, (0, 0.1), [0.0, v0], t_eval=np.linspace(0, 0.1, int(n_eval)),
                                 events=separation)
            x, v = solution.y
            x_pos = np.clip(x, 0, None)
            force = np.where(x <= 0, 0.0, IMPACT_STIFFNESS * x_pos ** 1.5 + chi * x_pos ** 1.5 * v)
            n = len(x)
            records.append(np.column_stack([x, v, 0.001 * force, np.full(n, e), np.full(n, v0)]))
            times.append(solution.t)
    return records, times
