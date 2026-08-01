# Çelikkubbe PC ↔ MCU Protocol Specification

**Version 1.0 — draft for electronics team sign-off**

Link: PC (Ubuntu 22.04, Python) ↔ ST Nucleo-G431KB, over RS422 (MAX490 differential, Waveshare isolated USB converter, Cat 5E, ~10 m).

This document is the contract between the software and electronics teams. Both firmware and `io/` are implemented from it. Changes require agreement from both sides.

---

## 1. Physical layer

| Parameter | Value |
|---|---|
| Baud rate | 921600 |
| Frame | 8N1, no flow control |
| Electrical | RS422 differential, full duplex |
| Cable | Cat 5E shielded, ~10 m |
| PC device | `/dev/ttyUSB*` (Waveshare converter) |

921600 baud is comfortable on 10 m of differential Cat 5E. Total load is under 10% of available bandwidth, leaving headroom for retransmission and debug output.

The Nucleo's ST-Link virtual COM port is reserved for firmware debug logging. Never mix debug text into the RS422 link.

---

## 2. Pin assignment

Corrected from the KiCad schematic (Rev 2026-08-06). **Two pins are swapped relative to that revision** — see the safety note below.

| Function | Pin | Direction | Notes |
|---|---|---|---|
| RS422 TX | PA9 (USART1_TX) | out | via MAX490 |
| RS422 RX | PA10 (USART1_RX) | in | via MAX490 |
| PUL1 (pan) | PA12 | out | via 74HCT245 → JK-HSD86 |
| DIR1 (pan) | PB0 | out | via 74HCT245 |
| PUL2 (tilt) | PA8 | out | via 74HCT245 → JK-HSD57 |
| DIR2 (tilt) | PA11 | out | via 74HCT245 |
| ALM1 (pan) | PA0 | in | open collector, needs pull-up |
| ALM2 (tilt) | PA1 | in | open collector, needs pull-up |
| **E-STOP** | **PB4** | **in** | NO contact pulls to GND |
| **AIRSOFT trigger** | **PA5** | **out** | → 220 Ω → PC817 |
| FAN PWM 1/2/3 | PB5, PB6, PB7 | out | |
| FAN TACH 1/2/3 | PA6, PA7, PB8 | in | open collector, needs pull-up |

### Safety note — why PB4 and PA5 are swapped

On STM32G4, **PB4 is NJTRST** and comes out of reset in JTAG alternate-function mode with an internal pull-up active. The pin sits high from power-on until firmware reconfigures it.

Driving the airsoft trigger from PB4 therefore energises the PC817 LED on every power-up, reset and firmware flash. The internal pull-up is weak (~40 kΩ, roughly 50 µA), which is probably below the optocoupler's threshold — but "probably" is not an acceptable guarantee for a fire-control output, and an attached debugger drives NJTRST actively, removing even that margin. The stated design principle is that the firing mechanism defaults to no-fire; a pin that floats high at reset violates it.

The same pull-up is ideal for the E-stop input: the NO contact pulls the pin to GND when pressed and the internal pull-up holds it high otherwise, with no external resistor needed.

**Additional hardware requirement:** fit a 10 kΩ pull-down from the PC817 LED anode to GND, so the trigger stays inactive even when the MCU pin is high-impedance.

### Open hardware items

- ALM and TACH lines are open-collector with no pull-ups shown in the schematic. Enable STM32 internal pull-ups, or fit external 4.7 kΩ resistors. Confirm which.
- Driver ENA inputs are unwired, so software motor disable is unavailable. Accepted for now; the `MotorEnable` command is not in this protocol.
- No limit switches exist. Homing is operator-assisted via the `zero` command (section 5). Adding one switch per axis would allow automatic homing and is recommended.

---

## 3. Frame formats

The protocol is deliberately asymmetric.

**PC → MCU: newline-delimited JSON with a checksum.** Low rate (≤ 50 Hz), and being human-readable means commands can be sent from a serial terminal during bring-up. That is worth a great deal during integration week.

**MCU → PC: fixed binary.** High rate (100 Hz), fixed layout, parsed with a single `struct.unpack`.

### 3.1 PC → MCU

```
<json>*<crc16_hex>\n
```

CRC16-CCITT (poly 0x1021, init 0xFFFF) computed over the JSON bytes only, excluding `*`, uppercase hex, four characters. The pattern follows NMEA 0183.

```
{"cmd":"aim","az":123.45,"el":-12.30,"seq":1042}*A3F2\n
```

Rules:

- `seq` is a monotonically increasing `uint16`, wrapping at 65535. Present on every command.
- Maximum frame length 256 bytes. Longer frames are discarded.
- A frame failing CRC is discarded silently and counted in `crc_error_count`.
- A duplicate `seq` is discarded (idempotent retransmission).

### 3.2 MCU → PC

```
0xAA 0x55 | LEN(1) | TYPE(1) | SEQ(2) | PAYLOAD(LEN) | CRC16(2)
```

Little-endian throughout. `LEN` is the payload length only. CRC16-CCITT over `LEN`, `TYPE`, `SEQ` and `PAYLOAD`.

| TYPE | Name | Rate |
|---|---|---|
| 0x80 | TELEMETRY | 100 Hz |
| 0x81 | ACK | per command |
| 0x82 | EVENT | asynchronous |
| 0x83 | LOG | ≤ 10 Hz |

---

## 4. Telemetry payload (TYPE 0x80)

```c
typedef struct __attribute__((packed)) {
    uint32_t mcu_ms;            // MCU uptime, diagnostics only
    float    pan_deg;           // COMMANDED position (step integral)
    float    tilt_deg;          // COMMANDED position
    float    pan_vel_dps;
    float    tilt_vel_dps;
    float    target_pan_deg;    // active setpoint echo
    float    target_tilt_deg;
    uint16_t status;            // bitfield, see below
    uint16_t fan_rpm[3];
    int16_t  mcu_temp_c10;      // deci-degrees Celsius
    uint16_t loop_time_us;
    uint16_t crc_error_count;
    uint16_t ammo_fired;        // shots since last arm
} telemetry_t;                  // 44 bytes, 52 on the wire
```

At 100 Hz this is 5.2 kB/s, about 5.6% of the link.

### Position is commanded, not measured

The motor encoders wire to the stepper drivers, not to the MCU. The drivers close the position loop internally, so `pan_deg` and `tilt_deg` are integrals of issued step pulses. If a driver faults or loses steps beyond its own correction range, these values drift from reality with no way for the MCU to detect it. The only external evidence is the ALM line.

This is why `target_pan_deg` / `target_tilt_deg` echo the active setpoint: the PC uses that echo to confirm its command was received before treating an aim as settled.

### Status bitfield

| Bit | Meaning |
|---|---|
| 0 | `armed` |
| 1 | `estop_active` |
| 2 | `position_valid` |
| 3 | `motion_complete` |
| 4 | `driver_alarm_pan` |
| 5 | `driver_alarm_tilt` |
| 6 | `homed_pan` |
| 7 | `homed_tilt` |
| 8 | `watchdog_tripped` |
| 9 | `limit_pan` |
| 10 | `limit_tilt` |
| 11–13 | `mcu_mode`: 0 BOOT, 1 IDLE, 2 READY, 3 MOVING, 4 SAFE |
| 14–15 | reserved |

`position_valid` clears whenever the E-stop trips or a driver alarm fires. The 48 V rail is cut on E-stop, and an unpowered stepper only holds detent torque — roughly 0.05–0.1 Nm against a tilt gravity torque of 0.78 Nm — so the tilt axis droops and the step integral no longer matches physical position. The PC must refuse to fire and require re-homing until this bit is set again.

---

## 5. Commands (PC → MCU)

| `cmd` | Fields | Effect |
|---|---|---|
| `hb` | `seq` | Heartbeat. 20 Hz. Resets the watchdog. |
| `mode` | `m`: `"idle"` \| `"ready"` \| `"safe"` | MCU mode transition |
| `aim` | `az`, `el`, `vmax`, `amax` | Absolute setpoint in degrees; velocity dps, acceleration dps² |
| `jog` | `axis`: `"pan"`\|`"tilt"`, `dir`: ±1, `speed` | Continuous motion while heartbeats continue |
| `stop` | — | Decelerate both axes to rest |
| `home` | `axes`: `"pan"`\|`"tilt"`\|`"both"` | Reserved; needs limit switches |
| `zero` | `axis`, `value` | Declare the current position to be `value` degrees |
| `arm` | — | Enable the fire chain |
| `disarm` | — | Disable the fire chain |
| `fire` | `n` | Fire `n` shots. Rejected unless armed. |
| `estop` | — | Soft e-stop: stop motion, disarm, enter SAFE |
| `param` | `id`, `v` | Runtime parameter write |

### Aim behaviour and backlash

`aim` moves to an absolute angle with a trapezoidal velocity profile. Because there is no output-side position feedback, gear backlash is uncorrectable in software — roughly 0.1° at the pan output for a module-5 spur pair.

The firmware compensates mechanically: **the final approach is always made from the same direction.** If the commanded move would arrive from the wrong side, overshoot by `BACKOFF_DEG` (0.5°) and approach from the consistent side. Backlash then always settles the same way and cancels out. This costs about 0.2 s per engagement and roughly halves the effective pan error.

`motion_complete` asserts when the trajectory generator finishes, not when position is measured — no measurement exists.

### Homing without switches

There are no reference sensors, so absolute position is undefined at power-up. Interim procedure: the operator centres the turret by eye, then the GUI sends `zero` for each axis. `homed_pan` / `homed_tilt` assert on receipt.

The PC refuses to enter M3 OPERATIONAL until both are set.

### Fire behaviour

`fire` is rejected unless `armed` is set, the E-stop is clear and `position_valid` holds. Rejections return a NACK with a reason code — the MCU is an independent safety authority and does not rely on the PC having checked first.

Each shot produces a trigger pulse of `param` `trigger_pulse_ms` (default 60 ms) with a minimum inter-shot gap of `trigger_gap_ms` (default 200 ms).

---

## 6. ACK (TYPE 0x81)

```c
typedef struct __attribute__((packed)) {
    uint16_t ack_seq;    // seq of the command being acknowledged
    uint8_t  result;     // 0 = OK, non-zero = rejection reason
    uint8_t  detail;
} ack_t;
```

| `result` | Meaning |
|---|---|
| 0 | Accepted |
| 1 | Bad CRC |
| 2 | Malformed JSON |
| 3 | Unknown command |
| 4 | Parameter out of range |
| 5 | Rejected in current mode |
| 6 | Not armed |
| 7 | E-stop active |
| 8 | Position not valid |
| 9 | Driver alarm |
| 10 | Software limit exceeded |

Every command is acknowledged. `hb` is acknowledged implicitly by the next telemetry frame rather than by an explicit ACK.

---

## 7. EVENT (TYPE 0x82)

Asynchronous notifications. The PC must not depend on polling to learn about these.

```c
typedef struct __attribute__((packed)) {
    uint8_t  event_id;
    uint8_t  axis;       // 0 pan, 1 tilt, 255 not applicable
    uint32_t mcu_ms;
    float    value;
} event_t;
```

| `event_id` | Meaning |
|---|---|
| 1 | E-stop pressed |
| 2 | E-stop released |
| 3 | Driver alarm asserted |
| 4 | Driver alarm cleared |
| 5 | Watchdog tripped |
| 6 | Motion complete |
| 7 | Shot fired |
| 8 | Software limit hit |
| 9 | Fan fault (tacho out of range) |
| 10 | Over-temperature |

---

## 8. Watchdog and failure behaviour

| Condition | MCU response | PC response |
|---|---|---|
| No valid command for 200 ms | Stop motion, disarm, enter SAFE, set `watchdog_tripped` | — |
| No telemetry for 300 ms | — | Declare link lost, transition to M4 SAFE |
| Driver alarm | Stop motion, disarm, clear `position_valid`, emit EVENT | Transition to M4 |
| E-stop pressed | 48 V cut in hardware; MCU disarms, clears `position_valid`, emits EVENT | Transition to M4, require re-homing |
| CRC error | Discard frame, increment counter | Log; warn above a threshold rate |

The two timeouts differ deliberately. 200 ms is the MCU's own budget for safing itself; the PC waits 300 ms so it does not race the MCU and drop to SAFE on ordinary scheduling jitter.

The E-stop remains a hardware interlock. Its NC contact cuts the 48 V rail regardless of firmware state; the NO contact is only how the MCU learns it happened.

---

## 9. Startup sequence

1. PC opens the port, sends `{"cmd":"mode","m":"idle"}`
2. MCU replies with ACK, begins 100 Hz telemetry
3. PC verifies `mcu_mode == IDLE`, alarms clear, fans turning
4. PC begins 20 Hz heartbeats
5. Operator centres the turret; PC sends `zero` for each axis
6. PC sends `{"cmd":"mode","m":"ready"}`
7. System is ready for operational mode

---

## 10. Rate summary

| Direction | Message | Rate | Bytes/s |
|---|---|---|---|
| PC → MCU | heartbeat | 20 Hz | ~800 |
| PC → MCU | aim | ≤ 30 Hz | ~2400 |
| MCU → PC | telemetry | 100 Hz | 5200 |
| MCU → PC | ack, event, log | variable | < 500 |

Total under 9 kB/s against roughly 92 kB/s available.

---

## Sign-off

| Team | Name | Date |
|---|---|---|
| Software | | |
| Electronics | | |

Changes after sign-off require agreement from both teams and a version bump.
