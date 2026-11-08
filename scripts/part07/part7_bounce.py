"""One ground impact followed by a second airborne leg, in SI units."""

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from scripts.common.still_air import Motion, Physics, Settings, _integrate, _rhs, inside_goal, spin_at


TIE_TOL_S = 1e-8


@dataclass
class Impact:
    time: float
    position: NDArray
    v_minus: NDArray
    v_plus: NDArray
    omega_minus: NDArray
    omega_plus: NDArray
    j_n: float
    j_t: NDArray
    slip_minus: NDArray
    slip_plus: NDArray
    energy_loss: float


@dataclass
class Segment:
    times: NDArray
    states: NDArray
    omega_end: NDArray


@dataclass
class BounceFlight:
    event: str
    time: float
    state: NDArray
    segments: tuple[Segment, ...]
    impact: Impact | None = None
    tie_gap: float | None = None  # t_goal - t_ground on the same free-flight leg
    message: str = ''


def ground_impact(time, state, omega, physics: Physics, e_n, mu_f):
    """Apply a normal restitution law and a Coulomb impulse capped at zero slip."""
    if not np.isfinite(e_n) or not 0 <= e_n <= 1 or not np.isfinite(mu_f) or mu_f < 0:
        raise ValueError('Require 0 <= e_n <= 1 and finite mu_f >= 0.')
    position = np.array(state[:3], dtype=float)
    v_minus = np.asarray(state[3:], dtype=float)
    omega = np.asarray(omega, dtype=float)
    if v_minus[2] >= 0:
        raise ValueError('Impact requires downward incident velocity.')

    position[2] = physics.radius
    mass = physics.mass
    radius = physics.radius
    inertia = (2 / 3) * mass * radius**2
    arm = np.array([0., 0., -radius])
    slip = v_minus + np.cross(omega, arm)
    slip[2] = 0.
    slip_speed = np.linalg.norm(slip)
    effective_mass = 1 / (1 / mass + radius**2 / inertia)
    j_n = -mass * (1 + e_n) * v_minus[2]
    j_t = np.zeros(3)
    if slip_speed > 0:
        j_t = -min(mu_f * j_n, effective_mass * slip_speed) * (slip / slip_speed)

    v_plus = v_minus + j_t / mass
    v_plus[2] = -e_n * v_minus[2]
    omega_plus = omega + np.cross(arm, j_t) / inertia
    slip_plus = v_plus + np.cross(omega_plus, arm)
    slip_plus[2] = 0.
    normal_loss = .5 * mass * (1 - e_n**2) * v_minus[2]**2
    tangent_loss = -j_t @ slip - (j_t @ j_t) / (2 * effective_mass)
    return Impact(time, position, v_minus, v_plus, omega, omega_plus,
                  float(j_n), j_t, slip, slip_plus, float(normal_loss + tangent_loss))


def _other_root(rhs, t_end, state, t_limit, settings, is_goal):
    """Probe the other event on the same unreset path over at most 10 ns."""
    if is_goal:
        index, level, direction = 2, 0., -1
    else:
        index, level, direction = 0, settings.goal_x, 1
    gap = state[index] - level
    rate = state[index + 3]
    if direction * gap >= 0:
        # At an exact tie a root state can lie a few ulps beyond the other surface.
        if direction * rate > 0 and abs(gap / rate) <= 32 * np.finfo(float).eps * max(1., t_end):
            return 0., ''
        return None, ''
    duration = min(TIE_TOL_S, t_limit - t_end)
    if duration <= 0:
        return None, ''

    def event(_time, trial):
        return trial[index] - level

    event.direction = direction
    event.terminal = True
    probe = _integrate(lambda dt, trial: rhs(t_end + dt, trial), state, duration, settings,
                       events=event)
    if not probe.success:
        return None, probe.message
    if probe.t_events[0].size and probe.t_events[0][0] < TIE_TOL_S:
        elapsed = float(probe.t_events[0][0])
        return (-elapsed if is_goal else elapsed), ''
    return None, ''


def _leg(initial, motion, physics, settings, start):
    # Use height above contact so short rebounds do not lose precision to z - R.
    # The shared RHS depends on velocity and spin only, so this translation is exact.
    local = initial.copy()
    local[2] -= physics.radius
    rhs = _rhs(motion, physics)
    t_limit = settings.t_max - start
    on_surface = local[2] == 0.

    def goal(_time, state):
        return state[0] - settings.goal_x

    def ground(time, state):
        if on_surface:
            # h(t)/t has the same returning root and its launch limit is positive.
            return state[2] / time if time > 0 else local[5]
        return state[2]

    goal.direction, ground.direction = 1, -1
    goal.terminal = ground.terminal = True
    sol = _integrate(rhs, local, t_limit, settings, events=(goal, ground))
    end = float(sol.t[-1])
    times = sol.t + start
    if end == t_limit:
        times[-1] = settings.t_max
    states = sol.y.T.copy()
    states[:, 2] += physics.radius
    segment = Segment(times, states, spin_at(end, motion))
    if not sol.success:
        return segment, 'numerical_failure', None, sol.message
    is_goal, is_ground = (bool(events.size) for events in sol.t_events)
    if not (is_goal or is_ground):
        return segment, 'timeout', None, ''
    if is_ground:
        states[-1, 2] = physics.radius
        if states[-1, 5] >= 0. or times[-1] <= start:
            return segment, 'numerical_failure', None, 'Returning contact is below time resolution.'
    if is_goal:
        states[-1, 0] = settings.goal_x
    tie, failure = _other_root(rhs, end, sol.y[:, -1], t_limit, settings, is_goal)
    if failure:
        return segment, 'numerical_failure', None, failure
    if tie is not None:
        return segment, 'goal_ground_tie', tie, ''
    if is_ground:
        return segment, 'ground_first', None, ''
    event = 'goal_inside' if inside_goal(*states[-1, 1:3], settings) else 'goal_outside'
    return segment, event, None, ''


def simulate_bounce(r0, motion: Motion, physics=Physics(), settings=Settings(),
                    *, e_n=.6, mu_f=0., allow_bounce=True, post_impact_magnus=True):
    """Integrate to the first goal-plane crossing or the last supported contact.

    Segment times are measured from release. At impact the last state of the
    first leg and the first state of the next leg have the same position/time.
    """
    initial = np.r_[r0, motion.v0].astype(float)
    if (np.shape(r0) != (3,) or np.shape(motion.v0) != (3,)
            or not np.all(np.isfinite(initial))):
        raise ValueError('Release position and velocity must be finite 3-vectors.')
    if initial[2] < physics.radius or initial[0] >= settings.goal_x:
        raise ValueError('Release must be at or above contact height and before the goal plane.')
    if motion.cd < 0 or not motion.tau > 0 or not np.isfinite(motion.cd + motion.k_l):
        raise ValueError('Require finite coefficients, nonnegative drag and positive decay time.')
    if np.shape(motion.omega0) != (3,) or not np.all(np.isfinite(motion.omega0)):
        raise ValueError('Angular velocity must be a finite 3-vector.')
    if not np.isfinite(e_n) or not 0 <= e_n <= 1 or not np.isfinite(mu_f) or mu_f < 0:
        raise ValueError('Require 0 <= e_n <= 1 and finite mu_f >= 0.')
    if initial[2] == physics.radius and initial[5] <= 0.:
        return BounceFlight('initial_ground', 0., initial, ())

    segment, event, tie, message = _leg(initial, motion, physics, settings, 0.)
    time, state = float(segment.times[-1]), segment.states[-1]
    if event != 'ground_first' or not allow_bounce:
        return BounceFlight(event, time, state, (segment,), tie_gap=tie, message=message)
    impact = ground_impact(time, state, segment.omega_end, physics, e_n, mu_f)
    restart = np.r_[impact.position, impact.v_plus]
    if e_n == 0.:
        return BounceFlight('no_liftoff', time, restart, (segment,), impact)
    if time >= settings.t_max:
        return BounceFlight('timeout', time, restart, (segment,), impact)

    rebound = replace(motion, v0=impact.v_plus, omega0=impact.omega_plus,
                      k_l=motion.k_l if post_impact_magnus else 0.)
    second, event, tie, message = _leg(restart, rebound, physics, settings, time)
    if event == 'ground_first':
        event = 'second_ground'
    return BounceFlight(event, float(second.times[-1]), second.states[-1],
                        (segment, second), impact, tie, message)
