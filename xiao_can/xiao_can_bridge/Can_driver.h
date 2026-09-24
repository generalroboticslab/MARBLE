#ifndef CAN_DRIVER_H
#define CAN_DRIVER_H

#include <SPI.h>
#include <mcp_can_dfs.h>
#include <mcp_canbus.h>

const int SPI_CS_PIN = D7; // D7 is standard CS for Seeed XIAO CAN Breakout
MCP_CAN CAN(SPI_CS_PIN);

bool init_can(byte speed = CAN_1000KBPS) {
  if (CAN.begin(speed) != CAN_OK) {
    return false;
  }
  // Configure MCP2515 RX masks to 0 to accept all CAN IDs (Standard & Extended)
  CAN.init_Mask(0, 0, 0x00000000);
  CAN.init_Mask(1, 1, 0x00000000);

  // Set all filters to 0
  for (int i = 0; i < 6; i++) {
    CAN.init_Filt(i, 0, 0x00000000);
  }
  return true;
}

byte send_can_message_status(unsigned long id, byte len, const byte *buf) {
  return CAN.sendMsgBuf(id, (id > 0x7FF ? 1 : 0), len, (byte *)buf);
}

bool send_can_message(unsigned long id, byte len, const byte *buf) {
  return send_can_message_status(id, len, buf) == CAN_OK;
}

byte can_error_status() {
  return CAN.checkError();
}

bool read_can_message(unsigned long &id, byte &len, byte *buf) {
  if (CAN_MSGAVAIL == CAN.checkReceive()) {
    CAN.readMsgBuf(&len, buf);
    id = CAN.getCanId();
    return true;
  }
  return false;
}

#endif
