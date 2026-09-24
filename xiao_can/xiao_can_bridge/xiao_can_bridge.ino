#include "Can_driver.h"

const char *BRIDGE_VERSION = "BALLBOT_XIAO_BRIDGE_V3";

String inputBuffer = "";

void setup() {
  // Configure onboard RGB LED pins
  pinMode(LED_RED, OUTPUT);
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_BLUE, OUTPUT);

  // Turn all LEDs off initially (active-low logic)
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, HIGH);
  digitalWrite(LED_BLUE, HIGH);

  // Blue on startup
  digitalWrite(LED_BLUE, LOW);

  Serial.begin(115200);

  // Wait up to 3 seconds for Serial (USB CDC) host connection, but proceed
  // anyway for headless
  unsigned long startWait = millis();
  while (!Serial && (millis() - startWait < 3000)) {
    delay(10);
  }
  delay(2000);

  // Failure Pattern 2: CAN initialization failed (Blink Red 2 times, pause)
  if (!init_can()) {
    digitalWrite(LED_BLUE, HIGH); // Turn off startup blue
    if (Serial)
      Serial.println("Error Initializing MCP2515...");
    while (1) {
      digitalWrite(LED_RED, LOW);
      delay(200);
      digitalWrite(LED_RED, HIGH);
      delay(200);
      digitalWrite(LED_RED, LOW);
      delay(200);
      digitalWrite(LED_RED, HIGH);
      delay(1000);
    }
  }

  // Running correctly: Turn off blue/red, turn on green
  digitalWrite(LED_BLUE, HIGH);
  digitalWrite(LED_RED, HIGH);
  digitalWrite(LED_GREEN, LOW);

  if (Serial)
    Serial.println("CAN Init OK at 1Mbps");
  if (Serial)
    Serial.println(BRIDGE_VERSION);
  inputBuffer.reserve(64);

  // Clear faults, then disable, on each motor. No homing and no encoder zeroing
  // here: the host does both (ballbot_terminal.py home_and_center, set_pos_zero).
  // Must match MOTOR_IDS in src/ballbot_runtime.py.
  const uint8_t motors[] = {0x08, 0x07, 0x06};
  byte clear_buf[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB};
  byte disable_buf[8] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD};
  delay(100);
  for (uint8_t motor_id : motors) {
    send_can_message(0x100 | motor_id, 8, clear_buf);
    delay(2);
  }
  delay(10);
  for (uint8_t motor_id : motors) {
    send_can_message(0x100 | motor_id, 8, disable_buf);
    delay(2);
  }
}

void processSerialCommand(String line) {
  line.trim();
  if (line == "V") {
    Serial.print("V:");
    Serial.println(BRIDGE_VERSION);
    return;
  }
  // Bench-only bitrate probe; no host script sends it. "B:<kbps>" (1000, 500 or
  // 250) re-initialises the MCP2515 at that rate and sends one disable frame to
  // CAN ID 0x102 (motor 0x02, which is not on this robot). The bridge stays at
  // that rate until it is reset.
  if (line.startsWith("B:")) {
    int kbps = line.substring(2).toInt();
    byte speed = kbps == 1000 ? CAN_1000KBPS
                 : kbps == 500 ? CAN_500KBPS
                 : kbps == 250 ? CAN_250KBPS
                               : 0;
    bool initialized = speed != 0 && init_can(speed);
    byte tx_status = 0xFF;
    byte error_status = 0xFF;
    if (initialized) {
      byte disable_buf[8] = {0xFF, 0xFF, 0xFF, 0xFF,
                             0xFF, 0xFF, 0xFF, 0xFD};
      delay(100);
      tx_status = send_can_message_status(0x102, 8, disable_buf);
      error_status = can_error_status();
    }
    Serial.print("B:");
    Serial.print(kbps);
    Serial.print(":");
    Serial.print(initialized ? 1 : 0);
    Serial.print(":");
    Serial.print(tx_status);
    Serial.print(":");
    Serial.println(error_status);
    return;
  }
  if (line.startsWith("S:") || line.startsWith("T:")) {
    bool report_status = line.startsWith("T:");
    int idx1 = line.indexOf(':', 2);
    int idx2 = line.indexOf(':', idx1 + 1);
    if (idx1 != -1 && idx2 != -1) {
      unsigned long id = strtoul(line.substring(2, idx1).c_str(), NULL, 16);
      int dlc = line.substring(idx1 + 1, idx2).toInt();
      String hex = line.substring(idx2 + 1);
      byte buf[8] = {0};
      for (int i = 0; i < dlc && i < 8; i++) {
        String byteHex = hex.substring(i * 2, i * 2 + 2);
        buf[i] = strtoul(byteHex.c_str(), NULL, 16);
      }
      byte tx_status = send_can_message_status(id, dlc, buf);
      if (report_status) {
        Serial.print("T:");
        Serial.print(id, HEX);
        Serial.print(":");
        Serial.print(tx_status);
        Serial.print(":");
        Serial.println(can_error_status());
      }
    }
  }
}

void loop() {
  // 1. Forward CAN packets to Serial
  unsigned long id;
  byte len = 0;
  byte buf[8];
  if (read_can_message(id, len, buf)) {
    if (Serial) {
      Serial.print("R:");
      Serial.print(id, HEX);
      Serial.print(":");
      Serial.print(len);
      Serial.print(":");
      for (int i = 0; i < len; i++) {
        if (buf[i] < 16)
          Serial.print("0");
        Serial.print(buf[i], HEX);
      }
      Serial.println();
    }
  }

  // 2. Forward Serial packets to CAN (Non-blocking byte-by-byte reader)
  while (Serial && Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      processSerialCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}
