"""SimTurretLink: a TurretLink with real physics, not a stub.

Lets GUI development, testing and rehearsal proceed without the turret.
Implements the same independent-authority safety checks docs/protocol.md
assigns to the MCU (reject fire unless armed/clear/valid, reject aim
outside software limits, trip its own watchdog) so a PC-side logic bug
shows up here exactly as it would against real firmware.

``core.types.Telemetry`` only has slots for the commanded (motor-side)
pan/tilt angle -- see its docstring. This simulator additionally tracks
the *true* mechanical output angle (``true_pan_deg``/``true_tilt_deg``)
purely for its own physical realism; nothing on the wire can ever report
it, so it is exposed as an extra diagnostic attribute, not through
``poll()``.
"""

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

from celikkubbe.core import config
from celikkubbe.core.commands import (
    Arm,
    Command,
    Disarm,
    Fire,
    Goto,
    Home,
    Jog,
    SetMode,
    SetParam,
    SetVelocity,
    SoftEstop,
    Zero,
)
from celikkubbe.core.protocols import Clock
from celikkubbe.core.types import Axis, Telemetry
from celikkubbe.io.codec import AckResult, EventId

# Pan backlash for a module-5 spur pair (docs/protocol.md section 5). Tilt
# uses a belt reduction, not modelled here -- gear backlash is a pan-only
# phenomenon in this drivetrain.
_BACKLASH_DEG = 0.1

# Fixed, arbitrary-but-consistent final-approach direction: the firmware
# compensation's whole point is that this sign is always the same one.
_CONSISTENT_APPROACH_SIGN = 1.0

# Detent torque (~0.05-0.1 Nm) vs tilt gravity torque (~0.78 Nm) -- not a
# measured rate, just fast enough to visibly droop within a test's
# timescale (docs/protocol.md section 4).
_GRAVITY_DROOP_DPS = 15.0

_FAN_RPM = (3000, 3000, 3000)
_MCU_TEMP_C = 40.0
_LOOP_TIME_US = 500


def _profile_duration(d: float, vmax: float, amax: float) -> float:
    """Total time for a trapezoidal (or triangular) 1-D move of distance
    ``d`` (>= 0). Shared by ``_trapezoidal`` and ``_AxisMotion.update``'s
    leg-chaining, which needs it to carry spillover time into the next leg.
    """
    if d < 1e-9:
        return 0.0
    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc * t_acc
    if 2 * d_acc >= d:
        return 2 * math.sqrt(d / amax)
    d_cruise = d - 2 * d_acc
    return 2 * t_acc + d_cruise / vmax


def _trapezoidal(start: float, target: float, vmax: float, amax: float, t: float):
    """Exact position/velocity of a trapezoidal (or triangular, for short
    moves) 1-D profile at elapsed time ``t``. Closed-form rather than
    incremental Euler integration: exact regardless of how coarsely or
    unevenly ``t`` advances between calls, which matters under FakeClock,
    where a test can jump seconds in one step with no accumulated error.
    """
    distance = target - start
    d = abs(distance)
    if d < 1e-9:
        return target, 0.0, True
    if t <= 0.0:
        return start, 0.0, False
    sign = math.copysign(1.0, distance)

    total_t = _profile_duration(d, vmax, amax)
    if t >= total_t - 1e-9:  # tolerance for float accumulation over many small steps
        return target, 0.0, True

    t_acc = vmax / amax
    d_acc = 0.5 * amax * t_acc * t_acc
    if 2 * d_acc >= d:
        t_half = total_t / 2.0
        if t < t_half:
            pos_1d, vel_1d = 0.5 * amax * t * t, amax * t
        else:
            t2 = total_t - t
            pos_1d, vel_1d = d - 0.5 * amax * t2 * t2, amax * t2
    else:
        t_cruise = total_t - 2 * t_acc
        if t < t_acc:
            pos_1d, vel_1d = 0.5 * amax * t * t, amax * t
        elif t < t_acc + t_cruise:
            pos_1d, vel_1d = d_acc + vmax * (t - t_acc), vmax
        else:
            t2 = total_t - t
            pos_1d, vel_1d = d - 0.5 * amax * t2 * t2, amax * t2
    return start + sign * pos_1d, sign * vel_1d, False


@dataclass
class _AxisMotion:
    """One axis's trapezoidal motion, with an optional queue of further
    legs to chain once the current one finishes (pan's backlash-overshoot
    compensation needs two legs; tilt only ever uses one).
    """

    position_deg: float = 0.0
    velocity_dps: float = 0.0
    frozen: bool = False
    _leg_start_t: float = 0.0
    _leg_start_deg: float = 0.0
    _leg_target_deg: float = 0.0
    _leg_vmax: float = 1.0
    _leg_amax: float = 1.0
    _pending_targets: list[float] = field(default_factory=list)

    def start_move(
        self, now: float, target_deg: float, vmax: float, amax: float, extra_legs=()
    ) -> None:
        self._leg_start_t = now
        self._leg_start_deg = self.position_deg
        self._leg_target_deg = target_deg
        self._leg_vmax = max(vmax, 1e-6)
        self._leg_amax = max(amax, 1e-6)
        self._pending_targets = list(extra_legs)

    def update(self, now: float) -> bool:
        """Advance to ``now``; returns True once every queued leg is done."""
        if self.frozen:
            self.velocity_dps = 0.0
            return False
        elapsed = now - self._leg_start_t
        pos, vel, done = _trapezoidal(
            self._leg_start_deg, self._leg_target_deg, self._leg_vmax, self._leg_amax, elapsed
        )
        self.position_deg = pos
        self.velocity_dps = vel
        if done and self._pending_targets:
            # A single feed()/poll() call can advance the clock past
            # several legs at once (a FakeClock jump, or just a slow
            # caller) -- carry the time this leg overran by into the next
            # leg's own clock instead of restarting it at zero, or a big
            # enough jump would silently freeze mid-chain.
            total_t = _profile_duration(
                abs(self._leg_target_deg - self._leg_start_deg), self._leg_vmax, self._leg_amax
            )
            excess = elapsed - total_t
            next_target = self._pending_targets.pop(0)
            pending = self._pending_targets
            self.start_move(now, next_target, self._leg_vmax, self._leg_amax, pending)
            self._leg_start_t = now - excess
            return self.update(now)
        return done and not self._pending_targets


class _Backlash:
    """Tracks the true mechanical output angle behind a motor position that
    can move without the output moving, for ``_gap`` degrees, whenever the
    motor's direction of travel reverses -- the actual physical phenomenon
    ``_AxisMotion``'s trapezoidal profile alone does not model (that class
    only computes the *commanded* position, which is all the wire protocol
    can ever report; see the module docstring).
    """

    def __init__(self, gap_deg: float) -> None:
        self._gap = gap_deg
        self.true_deg = 0.0
        self._motor_deg = 0.0
        self._direction = 0
        self._slack_remaining = 0.0

    def drive_to(self, motor_deg: float) -> None:
        delta = motor_deg - self._motor_deg
        self._motor_deg = motor_deg
        if delta == 0.0:
            return
        direction = 1 if delta > 0 else -1
        if direction != self._direction and self._direction != 0:
            self._slack_remaining = self._gap
        self._direction = direction
        travel = abs(delta)
        used = min(travel, self._slack_remaining)
        self._slack_remaining -= used
        self.true_deg += math.copysign(travel - used, delta)

    def reset(self, deg: float) -> None:
        self.true_deg = deg
        self._motor_deg = deg
        self._direction = 0
        self._slack_remaining = 0.0


class SimTurretLink:
    """A ``TurretLink`` backed by a physics simulation instead of hardware.

    Extra attributes beyond the ``TurretLink`` Protocol (``core`` never
    sees or depends on these; they exist for tests, a future GUI debug
    panel, and ``link_worker``'s diagnostics):

    ``true_pan_deg`` / ``true_tilt_deg``   -- ground-truth output angle
    ``ammo_fired``                         -- shots actually accepted
    ``last_ack``                           -- AckResult of the last command
    ``watchdog_tripped``                   -- sim's own 200ms watchdog
    ``crc_error_count``                    -- injected-fault counter
    ``pending_events`` / ``drain_events()``-- EVENT-equivalent notifications
    ``pending_acks`` / ``drain_acks()``    -- (seq, AckResult) pairs, for
                                               link_worker's ACK tracking
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._pan = _AxisMotion()
        self._tilt = _AxisMotion()
        self._pan_backlash = _Backlash(_BACKLASH_DEG)
        self._armed = False
        self._estop = False
        self._driver_alarm_pan = False
        self._driver_alarm_tilt = False
        self._homed_pan = False
        self._homed_tilt = False
        now = clock.now()
        self._last_update_t = now
        self._last_contact_t = now
        self.watchdog_tripped = False
        self.ammo_fired = 0
        self.last_ack = AckResult.OK
        self.crc_error_count = 0
        self.pending_events: list[tuple[EventId, Axis | None, float]] = []
        self.pending_acks: list[tuple[int, AckResult]] = []
        self._outbound_seq = 0
        self._last_applied_seq: int | None = None

        self._latency_s = 0.0
        self._pending_commands: deque[tuple[float, int, Command]] = deque()
        self._dropout_until: float | None = None
        self._crc_error_rate = 0.0

    # --- TurretLink protocol ---

    @property
    def connected(self) -> bool:
        return True

    def send(self, cmd: Command) -> None:
        self.send_tracked(cmd)

    def send_tracked(self, cmd: Command, seq: int | None = None) -> int:
        """Like ``send``, but returns the seq used and records an ack for
        it in ``pending_acks`` -- link_worker needs the seq to correlate a
        later ACK/retransmit; ``TurretLink.send()`` alone has no way to
        report it. Passing ``seq`` explicitly (a retransmit) reuses it
        instead of allocating a fresh one, so protocol section 3.1's
        duplicate-seq dedup below actually has something to compare against.
        """
        use_seq = self._outbound_seq if seq is None else seq
        now = self._clock.now()
        self._last_contact_t = now
        if self._latency_s <= 0.0:
            self._dispatch(now, use_seq, cmd)
        else:
            self._pending_commands.append((now + self._latency_s, use_seq, cmd))
        if seq is None:
            self._outbound_seq = (self._outbound_seq + 1) & 0xFFFF
        return use_seq

    def _dispatch(self, now: float, seq: int, cmd: Command) -> None:
        if seq == self._last_applied_seq:
            # Duplicate retransmission: docs/protocol.md section 3.1 says
            # the MCU discards it -- ack it again without re-executing, so
            # a retransmitted Fire whose original ACK was merely lost in
            # transit does not fire a second time.
            self.pending_acks.append((seq, self.last_ack))
            return
        self._apply_command(now, cmd)
        self._last_applied_seq = seq
        self.pending_acks.append((seq, self.last_ack))

    def drain_acks(self) -> list[tuple[int, AckResult]]:
        acks, self.pending_acks = self.pending_acks, []
        return acks

    def send_heartbeat(self) -> None:
        """Not a Command -- resets the watchdog with no positional effect."""
        self._last_contact_t = self._clock.now()

    def poll(self) -> Telemetry | None:
        now = self._clock.now()
        self._drain_pending_commands(now)
        self._check_watchdog(now)
        self._advance_physics(now)
        self._last_update_t = now

        if self._dropout_until is not None and now < self._dropout_until:
            return None
        if self._crc_error_rate > 0.0 and random.random() < self._crc_error_rate:
            self.crc_error_count += 1
            return None

        return Telemetry(
            t=now,
            pan_deg=self._pan.position_deg,
            tilt_deg=self._tilt.position_deg,
            pan_vel_dps=self._pan.velocity_dps,
            tilt_vel_dps=self._tilt.velocity_dps,
            # The active setpoint echo is whatever the *current* leg is
            # driving toward -- during a backlash-compensated overshoot
            # leg, that is the overshoot point, not the final target, so
            # AngleGate correctly keeps blocking until the final leg starts.
            target_pan_deg=self._pan._leg_target_deg,
            target_tilt_deg=self._tilt._leg_target_deg,
            motion_complete=not self._pan._pending_targets
            and not self._tilt._pending_targets
            and self._pan.velocity_dps == 0.0
            and self._tilt.velocity_dps == 0.0,
            armed=self._armed,
            estop=self._estop,
            position_valid=(
                not self._estop and not self._driver_alarm_pan and not self._driver_alarm_tilt
            ),
            driver_alarm_pan=self._driver_alarm_pan,
            driver_alarm_tilt=self._driver_alarm_tilt,
            fan_rpm=_FAN_RPM,
            mcu_temp_c=_MCU_TEMP_C,
            loop_time_us=_LOOP_TIME_US,
            crc_error_count=self.crc_error_count,
        )

    # --- command handling (independent safety authority) ---

    def _drain_pending_commands(self, now: float) -> None:
        while self._pending_commands and self._pending_commands[0][0] <= now:
            _, seq, cmd = self._pending_commands.popleft()
            self._dispatch(now, seq, cmd)

    def _apply_command(self, now: float, cmd: Command) -> None:
        if isinstance(cmd, Fire):
            self._handle_fire(cmd)
        elif isinstance(cmd, Goto):
            self._handle_goto(now, cmd)
        elif isinstance(cmd, Jog):
            self._handle_jog(cmd)
        elif isinstance(cmd, Zero):
            self._handle_zero(cmd)
        elif isinstance(cmd, Arm):
            self._handle_arm()
        elif isinstance(cmd, Disarm):
            self._armed = False
            self.last_ack = AckResult.OK
        elif isinstance(cmd, SoftEstop):
            self._trip_estop()
        elif isinstance(cmd, SetMode | Home | SetParam):
            self.last_ack = AckResult.OK  # accepted, no behaviour modelled
        elif isinstance(cmd, SetVelocity):
            self.last_ack = AckResult.UNKNOWN_COMMAND  # no wire equivalent -- see codec.py

    def _handle_fire(self, cmd: Fire) -> None:
        # Estop and driver alarm both also disarm as a side effect (see
        # _trip_estop / inject_driver_alarm), so checking "not armed" first
        # would mask the actual root cause behind a generic NOT_ARMED on
        # every rejection. Report the more specific reason.
        if self._estop:
            self.last_ack = AckResult.ESTOP_ACTIVE
        elif self._driver_alarm_pan or self._driver_alarm_tilt:
            self.last_ack = AckResult.DRIVER_ALARM
        elif not self._armed:
            self.last_ack = AckResult.NOT_ARMED
        else:
            self.ammo_fired += cmd.count
            self.last_ack = AckResult.OK
            self.pending_events.append((EventId.SHOT_FIRED, None, self._clock.now()))

    def _handle_arm(self) -> None:
        if self._estop:
            self.last_ack = AckResult.ESTOP_ACTIVE
        else:
            self._armed = True
            self.last_ack = AckResult.OK

    def _handle_goto(self, now: float, cmd: Goto) -> None:
        pan_lo, pan_hi = config.PAN_LIMIT_DEG
        tilt_lo, tilt_hi = config.TILT_LIMIT_DEG
        if not (pan_lo <= cmd.az_deg <= pan_hi) or not (tilt_lo <= cmd.el_deg <= tilt_hi):
            self.last_ack = AckResult.LIMIT_EXCEEDED
            self.pending_events.append((EventId.LIMIT_HIT, None, now))
            return

        current_pan = self._pan.position_deg
        first_target, remaining = self._plan_pan_legs(current_pan, cmd.az_deg)
        self._pan.start_move(now, first_target, cmd.max_vel_dps, cmd.max_accel_dps2, remaining)
        self._tilt.start_move(now, cmd.el_deg, cmd.max_vel_dps, cmd.max_accel_dps2)
        self.last_ack = AckResult.OK

    def _plan_pan_legs(self, current_deg: float, target_deg: float) -> tuple[float, list[float]]:
        if target_deg == current_deg:
            return target_deg, []
        direct_sign = math.copysign(1.0, target_deg - current_deg)
        if not config.UNIDIRECTIONAL_APPROACH or direct_sign == _CONSISTENT_APPROACH_SIGN:
            return target_deg, []
        overshoot = target_deg - _CONSISTENT_APPROACH_SIGN * config.BACKOFF_DEG
        return overshoot, [target_deg]

    def _handle_jog(self, cmd: Jog) -> None:
        # Continuous motion toward whichever limit the direction points at,
        # with an effectively-instant accel so it reads as constant
        # velocity -- until overridden by the next aim/jog/zero.
        axis = self._pan if cmd.axis is Axis.PAN else self._tilt
        lo, hi = config.PAN_LIMIT_DEG if cmd.axis is Axis.PAN else config.TILT_LIMIT_DEG
        speed = cmd.direction * cmd.speed_dps
        axis.start_move(self._clock.now(), hi if speed > 0 else lo, abs(speed) or 1e-6, 1e9)
        self.last_ack = AckResult.OK

    def _handle_zero(self, cmd: Zero) -> None:
        if cmd.axis is Axis.PAN:
            self._pan.position_deg = cmd.value_deg
            self._pan.velocity_dps = 0.0
            self._pan._pending_targets = []
            self._pan.start_move(self._clock.now(), cmd.value_deg, 1.0, 1.0)
            self._pan_backlash.reset(cmd.value_deg)
            self._homed_pan = True
        else:
            self._tilt.position_deg = cmd.value_deg
            self._tilt.velocity_dps = 0.0
            self._tilt._pending_targets = []
            self._tilt.start_move(self._clock.now(), cmd.value_deg, 1.0, 1.0)
            self._homed_tilt = True
        self.last_ack = AckResult.OK

    def _trip_estop(self) -> None:
        # The hardware E-stop button (inject_estop) and the software
        # `estop` command (SoftEstop) are electrically different -- the
        # hardware NC contact cuts the 48V rail, the software one does not
        # -- but the protocol document gives both the same severity ("enter
        # SAFE" / disarm, clear position_valid), so both converge here.
        was_active = self._estop
        self._estop = True
        self._armed = False
        if not was_active:
            self.pending_events.append((EventId.ESTOP_PRESSED, None, self._clock.now()))

    def _check_watchdog(self, now: float) -> None:
        elapsed_ms = (now - self._last_contact_t) * 1000.0
        if elapsed_ms > config.WATCHDOG_TIMEOUT_MS:
            if not self.watchdog_tripped:
                self.watchdog_tripped = True
                self.pending_events.append((EventId.WATCHDOG_TRIPPED, None, now))
            self._armed = False
            self._pan.frozen = True
            self._tilt.frozen = True
        else:
            self.watchdog_tripped = False
            self._pan.frozen = self._driver_alarm_pan
            self._tilt.frozen = self._driver_alarm_tilt

    def _advance_physics(self, now: float) -> None:
        if self._estop:
            dt = max(0.0, now - self._last_update_t)
            lo, _ = config.TILT_LIMIT_DEG
            self._tilt.position_deg = max(lo, self._tilt.position_deg - _GRAVITY_DROOP_DPS * dt)
            self._tilt.velocity_dps = 0.0
            self._pan.velocity_dps = 0.0
            return
        self._pan.update(now)
        self._tilt.update(now)
        self._pan_backlash.drive_to(self._pan.position_deg)

    # --- fault injection ---

    def inject_estop(self) -> None:
        self._trip_estop()

    def release_estop(self) -> None:
        if self._estop:
            self.pending_events.append((EventId.ESTOP_RELEASED, None, self._clock.now()))
        self._estop = False

    def inject_driver_alarm(self, axis: Axis) -> None:
        now = self._clock.now()
        if axis is Axis.PAN:
            self._driver_alarm_pan = True
        else:
            self._driver_alarm_tilt = True
        self._armed = False
        self.pending_events.append((EventId.DRIVER_ALARM_ASSERTED, axis, now))

    def clear_driver_alarm(self, axis: Axis) -> None:
        now = self._clock.now()
        if axis is Axis.PAN:
            self._driver_alarm_pan = False
        else:
            self._driver_alarm_tilt = False
        self.pending_events.append((EventId.DRIVER_ALARM_CLEARED, axis, now))

    def inject_link_dropout(self, duration_s: float) -> None:
        self._dropout_until = self._clock.now() + duration_s

    def inject_crc_errors(self, rate: float) -> None:
        self._crc_error_rate = max(0.0, min(1.0, rate))

    def set_latency(self, ms: float) -> None:
        self._latency_s = max(0.0, ms) / 1000.0

    def drain_events(self) -> list[tuple[EventId, Axis | None, float]]:
        events, self.pending_events = self.pending_events, []
        return events

    # --- diagnostics: ground-truth position, invisible to any real telemetry ---

    @property
    def true_pan_deg(self) -> float:
        return self._pan_backlash.true_deg

    @property
    def true_tilt_deg(self) -> float:
        return self._tilt.position_deg
