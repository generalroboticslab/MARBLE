# Bring-up checklist

For a new or rebuilt robot. Work in order; do not free-roll until every
earlier section passes. Run commands from the repository root.

## 1. Runtime and firmware

- `./scripts/prepare_orangepi.sh`, then `./run.sh` option 1: only the
  `calibration.json is absent` warning may remain.
- The bridge is flashed and verified ([`FLASH_XIAO.md`](FLASH_XIAO.md)).
- Startup sends `0xFB` (clear fault) then `0xFD` (disable) on `0x108`, `0x107`
  and `0x106`, never `0xFE` (encoder zero).
- Restrained, `python scripts/gl_bench.py read` shows fresh telemetry from
  `0x08`, `0x07` and `0x06` without motion. If one is missing,
  `python scripts/scan_can_motors.py` lists the IDs that answer.
- A launch without a TTY exits before opening hardware.

## 2. IMU transform

- Sphere secured, motor power off, reference pose (+Y forward, +Z up).
- `python scripts/calibrate_imu.py` (`./run.sh` option 2; `--viz` optional):
  base +Z points up and base +Y matches the marked forward direction.
- Rotate about each marked axis: no axis swaps or sign errors.
- On failure, check the TM171 quaternion convention first, then re-solve with
  `--capture-upright` ([`SETUP.md`](SETUP.md#imu-mounting)).

## 3. Slider zero and mapping

- Run `./run.sh` option 3 per [`SETUP.md`](SETUP.md#slider-calibration),
  sphere secured and moving masses visible.
- Each test motion moves the slider away from the 0 mm hardstop.
- `config/calibration.json` lists each motor and each of Slider-5/6/7 exactly
  once.

## 4. Homing and safety

- Restrained, `python src/ballbot_terminal.py --home-only` gets fresh
  telemetry before any motion, reaches the 0 mm hardstop without sustained
  impact, zeros, centres at 114 mm and disables.
- `python scripts/rehome_auto.py` pauses before zeroing and ends at 0 mm.
- With one motor's telemetry unplugged, arming is refused or the motors
  disable within 5 s.
- SPACE centres (a step, with smoothing off), Ctrl+C disables and quits, and
  X disables until restart.
- With no movement key, the sliders centre after 3 s and the motors disable
  after 120 s.
- With injected temperatures, the controller trips at 85 °C motor or drive and
  re-arms only below 80 °C, on a new movement command.

## 5. Geometric controller

- Lift or fixture the sphere so it cannot roll.
- Start `python src/ballbot_terminal.py --policy heuristic --heuristic-magnitude 0.02`.
  Raise the magnitude only after each direction is confirmed.
- W keeps the mass toward command-frame +X as the shell is rotated slowly; A
  targets +Y. WA, WD, SA and SD are normalised diagonals.
- Targets stay within travel and return to centre.

## 6. Learned policy

- Off-robot first: `./run.sh` option 6 and `scripts/policy_sim_check.py`
  ([`DEVELOPING.md`](DEVELOPING.md#offline-checks)).
- Launch the land default as in
  [`OPERATIONS.md`](OPERATIONS.md#learned-policy): explicit `--policy-path`,
  default limits.
- Under sustained slew at 30 rad/s, watch `Tdrv` and confirm the 85 °C trip.
- Joint order is Slider-5, Slider-6, Slider-7; output signs match simulation.
- Logged observations replayed off-robot through the same TorchScript give the
  same actions.
- Measure IMU-to-CAN latency and the achieved rate (`t_s` in the log).
  Training modelled 0 to 40 ms of sensor delay (`delay_max_lag: 2`) and no
  action delay.
- P centres for 1 s, switches controller, and resumes only while the movement
  command is still active.
- Short W, D, S and A runs in a clear area, with X and the power cutoff at
  hand.
- First water session tethered, with manual override ready.

## 7. Record results

Add to this file: date, hardware revision, calibration backup name, policy
md5, PyTorch version, inference P50/P95, serial ports, homing peak torque,
temperature behaviour, pass/fail notes.
