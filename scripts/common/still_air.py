"""Forward flight and three-position inverse prediction in SI units."""

from dataclasses import dataclass, replace
from math import exp, hypot, inf, pi
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Physics:
    radius: float = 0.11
    mass: float = 0.43
    gravity: float = 9.81
    density: float = 1.20
    viscosity: float = 1.81e-5
    wind: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def area(self):
        return pi * self.radius**2

    @property
    def volume(self):
        return 4 * pi * self.radius**3 / 3

    @property
    def effective_gravity(self):
        return self.gravity * (1 - self.density * self.volume / self.mass)


@dataclass(frozen=True)
class Settings:
    rtol: float = 1e-9
    atol: float = 1e-11
    max_step: float = 0.002
    t_max: float = 2.0
    goal_x: float = 11.0
    goal_half_width: float = 3.66
    goal_height: float = 2.44


@dataclass(frozen=True)
class Motion:
    v0: ArrayLike
    cd: float
    omega0: ArrayLike = (0.0, 0.0, 0.0)
    k_l: float = 1.0
    tau: float = inf


@dataclass
class Flight:
    event: str
    time: float
    states: NDArray
    times: NDArray
    message: str = ''
    _interpolant: Callable | None = None

    def sample(self, times):
        """Interpolate [x, y, z, vx, vy, vz] at times within the flight."""
        times = np.atleast_1d(times).astype(float)
        if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(times > self.time):
            raise ValueError('Sample times must be within the integrated flight.')
        if self._interpolant is None:
            return np.repeat(self.states[:1], len(times), axis=0)
        return self._interpolant(times).T


@dataclass
class FitAttempt:
    initial: NDArray
    parameters: NDArray
    motion: Motion
    residuals: NDArray
    success: bool
    status: int
    message: str
    evaluations: int
    active_mask: NDArray

    @property
    def sse(self):
        return float(np.sum(self.residuals**2))


@dataclass
class FitResult:
    best: FitAttempt | None
    attempts: list[FitAttempt]
    parameter_names: tuple[str, ...]
    scales: NDArray
    lower: NDArray
    upper: NDArray
    basis: NDArray


# Parameter bounds and characteristic scales for the numerical SVD.
PARAMETERS = ('vx', 'vy', 'vz', 'cd', 'omega2', 'omega3')
LOWER = np.array([10., -20., 0.1, 0.05, -100., -100.])
UPPER = np.array([45., 20., 20., 0.60, 100., 100.])
SCALES = np.array([30., 10., 10., 0.20, 50., 50.])
EXPANDED_LOWER = np.array([5., -30., 0.01, 0.01, -150., -150.])
EXPANDED_UPPER = np.array([55., 30., 30., 1., 150., 150.])


def spin_at(time, motion):
    return np.asarray(motion.omega0) * exp(-time / motion.tau)


def _air_force_terms(velocity, wind, drag_scale, lift_scale, omega, decay):
    """Return drag and Magnus accelerations as three-component tuples."""
    ux, uy, uz = (v - w for v, w in zip(velocity, wind))
    wx, wy, wz = omega
    q = hypot(ux, uy, uz)
    drag = -drag_scale * q
    lift = lift_scale * decay
    return ((drag * ux, drag * uy, drag * uz),
            (lift * (wy * uz - wz * uy),
             lift * (wz * ux - wx * uz),
             lift * (wx * uy - wy * ux)))


def aerodynamic_acceleration(time, velocity, motion, physics):
    factor = physics.density * physics.area / (2 * physics.mass)
    drag, lift = _air_force_terms(
        velocity, physics.wind, factor * motion.cd, factor * motion.k_l * physics.radius,
        motion.omega0, exp(-time / motion.tau),
    )
    return np.array(drag), np.array(lift)


def _rhs(motion, physics):
    factor = physics.density * physics.area / (2 * physics.mass)
    drag_scale = factor * motion.cd
    lift_scale = factor * motion.k_l * physics.radius
    wind = tuple(physics.wind)
    omega = tuple(motion.omega0)
    inverse_tau = 1 / motion.tau
    gravity = physics.effective_gravity

    def rhs(time, state):
        vx, vy, vz = state[3:]
        drag, lift = _air_force_terms(
            (vx, vy, vz), wind, drag_scale, lift_scale, omega, exp(-time * inverse_tau),
        )
        return np.array([vx, vy, vz, drag[0] + lift[0], drag[1] + lift[1],
                         drag[2] + lift[2] - gravity])

    return rhs


def _integrate(rhs, initial, t_end, settings, **kwargs):
    return solve_ivp(
        rhs, (0., t_end), initial, method='DOP853', rtol=settings.rtol,
        atol=settings.atol, max_step=settings.max_step, **kwargs,
    )


def inside_goal(y, z, settings):
    # A root state can differ from an exact edge by a few floating-point ulps.
    edge_tol = 1e-12
    return (abs(y) < settings.goal_half_width - edge_tol
            and edge_tol < z < settings.goal_height - edge_tol)


def simulate(r0, motion: Motion, physics=Physics(), settings=Settings()):
    """Propagate a release to the first goal-plane crossing or ground contact."""
    initial = np.r_[r0, motion.v0].astype(float)
    if initial.shape != (6,) or np.any(~np.isfinite(initial)):
        raise ValueError('Release position and velocity must be finite 3-vectors.')
    if initial[2] < physics.radius or initial[0] >= settings.goal_x:
        raise ValueError('Release must be at or above contact height and before the goal plane.')
    if motion.cd < 0 or motion.tau <= 0 or not np.isfinite(motion.cd + motion.k_l):
        raise ValueError('Require finite coefficients, nonnegative drag and positive spin-decay time.')
    if np.shape(motion.omega0) != (3,) or not np.all(np.isfinite(motion.omega0)):
        raise ValueError('Angular velocity must be a finite 3-vector.')
    if initial[2] == physics.radius and initial[5] <= 0:
        return Flight('ground_first', 0.0, initial[None, :], np.array([0.0]))

    def goal_event(_t, state):
        return state[0] - settings.goal_x

    def ground_event(time, state):
        if time == 0 and initial[2] == physics.radius:
            # Discard the launch root. After launch, seek only downward contact.
            return np.spacing(physics.radius)
        return state[2] - physics.radius

    goal_event.direction = 1
    ground_event.direction = -1
    goal_event.terminal = ground_event.terminal = True
    solution = _integrate(
        _rhs(motion, physics), initial, settings.t_max, settings,
        events=(goal_event, ground_event), dense_output=True,
    )
    event = 'timeout'
    if not solution.success:
        event = 'numerical_failure'
    elif solution.t_events[1].size:
        event = 'ground_first'
    elif solution.t_events[0].size:
        y, z, vz = solution.y[[1, 2, 5], -1]
        if z <= physics.radius + 1e-12 and vz < 0:
            event = 'ground_first'
        else:
            event = 'goal_inside' if inside_goal(y, z, settings) else 'goal_outside'
    return Flight(event, float(solution.t[-1]), solution.y.T, solution.t,
                  solution.message, solution.sol)


def spin_basis(chord):
    """Return the column matrix [e1, e2, e3] aligned with the observed chord."""
    chord = np.asarray(chord, dtype=float)
    length = np.linalg.norm(chord)
    if length == 0:
        raise ValueError('The first and last positions must differ for a spin fit.')
    e1 = chord / length
    reference = (1., 0., 0.) if abs(e1[2]) > 0.99 else (0., 0., 1.)
    e2 = np.cross(reference, e1)
    e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def baseline_initials(times, positions, gravity):
    times = np.asarray(times, dtype=float)[1:]
    displacements = np.asarray(positions, dtype=float)[1:] - positions[0]
    v_ca, acceleration = np.linalg.solve(
        np.column_stack([times, times**2 / 2]), displacements,
    )
    gravity_displacement = times[:, None]**2 / 2 * np.array([0., 0., -gravity])
    v_g = times @ (displacements - gravity_displacement) / (times @ times)
    return v_ca, v_g, acceleration


def position_residuals(times, positions, motion, physics, settings):
    """Evaluate the smooth ODE at both observation times for least squares."""
    solution = _integrate(
        _rhs(motion, physics), np.r_[positions[0], motion.v0], times[-1], settings,
        t_eval=times[1:],
    )
    if not solution.success or solution.y.shape[1] != 2:
        raise FloatingPointError('Integration failed within the observation window.')
    return solution.y[:3].T - positions[1:]


def fit_from_positions(times, positions, physics=Physics(), settings=Settings(), *,
                       cd=None, fit_spin=True, k_l=1.0, tau=inf, omega_parallel=0.0,
                       random_starts=0, seed=20261107, warm_start=None,
                       expanded=()):
    """Fit launch velocity and any free aerodynamic parameters to three positions.

    The first position is fixed. A fitted path must remain in flight and before
    the goal through the observation window. Each optimizer run is retained.
    """
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if (times.shape != (3,) or positions.shape != (3, 3)
            or np.any(~np.isfinite(times)) or np.any(~np.isfinite(positions))
            or times[0] != 0 or not 0 < times[1] < times[2] < settings.t_max):
        raise ValueError('Supply three finite 3-D positions at 0 < t1 < t2 < t_max after release.')
    if positions[0, 2] < physics.radius or positions[0, 0] >= settings.goal_x:
        raise ValueError('Release must be at or above contact height and before the goal plane.')
    basis = spin_basis(positions[2] - positions[0]) if fit_spin else np.eye(3)
    free = np.array([True, True, True, cd is None, fit_spin, fit_spin])
    names = tuple(np.array(PARAMETERS)[free])
    widen = np.isin(PARAMETERS, expanded)
    lower = np.where(widen, EXPANDED_LOWER, LOWER)[free]
    upper = np.where(widen, EXPANDED_UPPER, UPPER)[free]
    scales = SCALES[free]
    fixed = np.array([0., 0., 0., cd if cd is not None else 0., 0., 0.])

    def unpack(parameters):
        full = fixed.copy()
        full[free] = parameters
        omega = basis @ [omega_parallel if fit_spin else 0., *full[4:]]
        return Motion(full[:3], full[3], omega, k_l, tau)

    def residual(scaled):
        return position_residuals(times, positions, unpack(scaled * scales),
                                  physics, settings).ravel()

    v_ca, v_g, local_acceleration = baseline_initials(times, positions, physics.gravity)
    local_air = local_acceleration + [0., 0., physics.effective_gravity]
    u = v_ca + local_acceleration * times[1] - physics.wind
    q = np.linalg.norm(u)
    factor = physics.density * physics.area / (2 * physics.mass)
    cd_guess = 0.20
    omega_guess = np.zeros(3)
    if factor > 0 and q > 0:
        cd_guess = -(local_air @ u) / (factor * q**3)
        if fit_spin and k_l != 0:
            omega_guess = basis.T @ np.cross(u, local_air) / (factor * k_l * physics.radius * q**2)
    initials = [np.r_[v_g, 0.20, 0., 0.][free],
                np.r_[v_ca, 0.20, 0., 0.][free],
                np.r_[v_ca, cd_guess, omega_guess[1:]][free]]
    if warm_start is not None:
        warm_spin = basis.T @ warm_start.omega0
        initials.insert(0, np.r_[warm_start.v0, warm_start.cd, warm_spin[1:]][free])
    initials.extend(np.random.default_rng(seed).uniform(
        lower, upper, size=(random_starts, len(names)),
    ))
    attempts = []
    for initial in initials:
        initial = np.clip(initial, lower, upper)
        try:
            fit = least_squares(
                residual, initial / scales, bounds=(lower / scales, upper / scales),
                method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-13, max_nfev=150,
            )
        except FloatingPointError as error:
            attempts.append(FitAttempt(initial, initial, unpack(initial), np.full((2, 3), np.nan),
                                       False, -2, str(error), 0, np.zeros(len(names), dtype=int)))
            continue
        parameters = fit.x * scales
        attempts.append(FitAttempt(initial, parameters, unpack(parameters), fit.fun.reshape(2, 3),
                                   fit.success, fit.status, fit.message, fit.nfev, fit.active_mask))

    best = None
    check = replace(settings, t_max=times[-1])
    for attempt in sorted((item for item in attempts if item.success), key=lambda item: item.sse):
        flight = simulate(positions[0], attempt.motion, physics, check)
        if flight.event == 'timeout':
            best = attempt
            break
        attempt.success = False
        attempt.message = f'Invalid in observation window: {flight.event}.'
    return FitResult(best, attempts, names, scales, lower, upper, basis)
