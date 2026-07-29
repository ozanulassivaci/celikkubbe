# Celikkubbe

Decision layer for a pan/tilt turret that detects, tracks, and engages targets
for the TEKNOFEST air defence competition.

## Features

- Mode state machine (M1 INIT -> M2 STANDBY -> M3 OPERATIONAL -> M4 SAFE) with
  no automatic recovery out of the safe state.
- Engagement state machine (S1 SEARCH -> S6 ASSESS) covering manual (Stage 1)
  and fully autonomous (Stage 2/3) operation, including IFF handling.
- L1/L2/L3 detection cascade with an automatic recovery monitor that probes
  the primary pipeline while running on a fallback.
- Health monitors (inference latency, camera, link, detection, depth) driven
  by consecutive-failure counters rather than rolling averages, so a single
  slow frame cannot trigger a fallback.
- Safety/IFF/range/confidence/angle/limit gates that report every failing
  reason at once, not just the first.
- Threat prioritisation with hysteresis so the target list does not thrash at
  frame rate.

All modules are pure: state machines return `Command` objects instead of
executing them, and every time-dependent component takes an injectable
`Clock` so tests run without real delays.

## Tech stack

- Python 3.10
- NumPy (numeric work only)
- pytest / pytest-cov for tests and coverage
- ruff for linting

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

This package currently exposes only the core decision layer under
`celikkubbe.core`. It has no camera, serial, or GUI dependencies and cannot
drive real hardware on its own; it is meant to be imported by an outer
application layer that supplies frames, telemetry, and executes the
`Command` objects it returns.

Run the test suite:

```bash
pytest
```

## Project structure

```
src/celikkubbe/
└── core/
    ├── types.py         # enums and data contracts
    ├── commands.py      # command objects returned by state machines
    ├── protocols.py     # Clock / FrameSource / TurretLink interfaces
    ├── config.py        # every threshold used by the layer
    ├── clock.py         # Clock, SystemClock, FakeClock
    ├── modes.py         # M1-M4 mode machine
    ├── engagement.py    # S1-S6 engagement machine
    ├── cascade.py       # L1/L2/L3 selection and recovery monitor
    ├── health.py        # health monitors
    ├── gates.py         # safety / IFF / range / confidence / angle gates
    ├── priority.py      # threat scoring and ordering
    └── strings.py       # Turkish UI strings keyed by reason code
```

## Limitations

- No hardware I/O: camera capture, serial protocol, and vision inference are
  separate layers not yet implemented in this repository.
- No GUI.
- Multi-target tracking / data association (`tracking.py`) is not part of
  this layer; it consumes already-formed `Track` objects.
- Ballistic lead-angle computation is not implemented here.

## License

Not yet decided.
