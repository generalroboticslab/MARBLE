# Flashing the XIAO CAN bridge

The firmware in `xiao_can/xiao_can_bridge/` relays CAN at 1 Mbit/s to and from
USB serial. At boot it sends clear-fault and then disable to motors `0x08`,
`0x07` and `0x06`. The motor driver refuses to run unless the bridge reports
`BALLBOT_XIAO_BRIDGE_V3`.

> [!WARNING]
> Keep motor power off until step 5 passes.

You need a USB-C data cable and a computer with this repository.

## 1. Install arduino-cli and the board package

```bash
brew install arduino-cli                                                                 # macOS
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh  # Linux
```

```bash
arduino-cli config init --overwrite
arduino-cli config add board_manager.additional_urls \
  https://files.seeedstudio.com/arduino/package_seeeduino_boards_index.json
arduino-cli core update-index
arduino-cli core install Seeeduino:nrf52@1.1.10
```

1.1.9 or later works; earlier releases lack the Sense Plus board.

## 2. Install the CAN library

```bash
arduino-cli lib install mcp_canbus@1.0.0
```

If the index is unavailable, place
<https://github.com/Longan-Labs/Arduino_CAN_BUS_MCP2515> in
`~/Documents/Arduino/libraries/` (macOS) or `~/Arduino/libraries/` (Linux).

## 3. Find the port

```bash
arduino-cli board list
```

The XIAO shows up as, for example, `/dev/cu.usbmodem14201` (macOS) or
`/dev/ttyACM0` (Linux); this is `<PORT>` below. If it is missing, double-tap
RESET: the bootloader appears as a drive named `XIAO-SENSE`.

## 4. Compile and upload

From the repository root:

```bash
arduino-cli compile --fqbn Seeeduino:nrf52:xiaonRF52840SensePlus xiao_can/xiao_can_bridge
arduino-cli upload --fqbn Seeeduino:nrf52:xiaonRF52840SensePlus --port <PORT> xiao_can/xiao_can_bridge
```

Use this FQBN for a plain Sense too; the pins used (D7 chip select, SPI on D8
to D10, RGB LED) map identically.
In the Arduino IDE, the board is Tools > Board > Seeed nRF52 Boards > Seeed
XIAO nRF52840 Sense Plus.

## 5. Verify

With motor power still off (needs `pyserial` and `numpy`):

```bash
python scripts/verify_xiao_firmware.py        # add --motor-port <PORT> if autodetect fails
```

Expected: `PASS: bridge reports BALLBOT_XIAO_BRIDGE_V3. Firmware is up to date.`

| Message | Meaning |
|---|---|
| `FAIL: expected BALLBOT_XIAO_BRIDGE_V3, bridge said: ...BRIDGE_V2` | The upload did not take. Repeat step 4. |
| `XIAO bridge firmware is outdated; flash BALLBOT_XIAO_BRIDGE_V3` at robot start | This procedure was not done. |

LED: blue while starting, green once CAN is up, repeating double red blink if
the MCP2515 failed to initialise (check the SPI wiring and the CAN board's
power).

Put the XIAO back if you removed it, then reconnect motor power.

The motor IDs in `motors[]` in `xiao_can_bridge.ino` must match `MOTOR_IDS` in
`src/ballbot_runtime.py`. Change both together.
