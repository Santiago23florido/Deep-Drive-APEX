/*
  APEX - DFRobot WT61PC / SEN0386 UART bridge for Arduino Nano 33 IoT

  Wiring:
    WT61PC TXD -> Arduino D0 / RX1 (Serial1 RX, schematic net RX_MCU)
    WT61PC RXD -> Arduino D1 / TX1 (Serial1 TX, schematic net TX_MCU)
    WT61PC GND -> Arduino GND
    WT61PC VCC -> 3V3 or 5V according to the module label

  Nano 33 IoT uses 3.3 V logic and is not 5 V tolerant. If the WT61PC TXD line
  outputs 5 V, use a level shifter before connecting it to D0/RX1.

  USB Serial output to PC/Raspberry:
    ax,ay,az,gx,gy,gz

  Units:
    acceleration: m/s^2
    angular velocity: rad/s

  This sketch parses WT61PC frames directly:
    0x55 0x51 ... acceleration
    0x55 0x52 ... gyroscope
    0x55 0x53 ... angle
*/

#include <Arduino.h>

static constexpr unsigned long DEBUG_BAUD = 115200;
static constexpr unsigned long SENSOR_BAUD = 9600;
static constexpr unsigned long SERIAL_WAIT_TIMEOUT_MS = 2500;
static constexpr unsigned long STATUS_PERIOD_MS = 1000;
static constexpr float GRAVITY_MPS2 = 9.80665f;
static constexpr float DEG2RAD = 0.017453292519943295f;

static constexpr uint8_t FRAME_START = 0x55;
static constexpr uint8_t FRAME_ACC = 0x51;
static constexpr uint8_t FRAME_GYRO = 0x52;
static constexpr uint8_t FRAME_ANGLE = 0x53;
static constexpr uint8_t FRAME_LEN = 11;

struct Vector3f {
  float x;
  float y;
  float z;
};

uint8_t frameBuffer[FRAME_LEN];
uint8_t frameIndex = 0;
Vector3f acc = {0.0f, 0.0f, 0.0f};
Vector3f gyro = {0.0f, 0.0f, 0.0f};
Vector3f angle = {0.0f, 0.0f, 0.0f};
bool haveAcc = false;
bool haveGyro = false;
bool haveAngle = false;
unsigned long lastStatusMs = 0;
unsigned long sampleCount = 0;

bool isKnownFrameType(uint8_t value) {
  return value == FRAME_ACC || value == FRAME_GYRO || value == FRAME_ANGLE;
}

uint8_t checksum(const uint8_t *frame) {
  uint8_t sum = 0;
  for (uint8_t i = 0; i < 10; i++) {
    sum += frame[i];
  }
  return sum;
}

int16_t readSigned16LE(const uint8_t *frame, uint8_t lowByteIndex) {
  const uint16_t raw = static_cast<uint16_t>(frame[lowByteIndex]) |
                       (static_cast<uint16_t>(frame[lowByteIndex + 1]) << 8);
  return static_cast<int16_t>(raw);
}

Vector3f decodeVector(const uint8_t *frame, float scale) {
  return {
    readSigned16LE(frame, 2) * scale,
    readSigned16LE(frame, 4) * scale,
    readSigned16LE(frame, 6) * scale,
  };
}

bool readWt61pcFrame(uint8_t *outFrame) {
  while (Serial1.available() > 0) {
    const uint8_t byteIn = static_cast<uint8_t>(Serial1.read());

    if (frameIndex == 0) {
      if (byteIn != FRAME_START) {
        continue;
      }
      frameBuffer[0] = byteIn;
      frameIndex = 1;
      continue;
    }

    if (frameIndex == 1) {
      if (!isKnownFrameType(byteIn)) {
        frameIndex = 0;
        if (byteIn == FRAME_START) {
          frameBuffer[0] = byteIn;
          frameIndex = 1;
        }
        continue;
      }
      frameBuffer[1] = byteIn;
      frameIndex = 2;
      continue;
    }

    frameBuffer[frameIndex] = byteIn;
    frameIndex++;
    if (frameIndex < FRAME_LEN) {
      continue;
    }

    frameIndex = 0;
    if (checksum(frameBuffer) != frameBuffer[10]) {
      continue;
    }
    memcpy(outFrame, frameBuffer, FRAME_LEN);
    return true;
  }
  return false;
}

void processFrame(const uint8_t *frame) {
  switch (frame[1]) {
    case FRAME_ACC:
      acc = decodeVector(frame, (16.0f * GRAVITY_MPS2) / 32768.0f);
      haveAcc = true;
      break;
    case FRAME_GYRO:
      gyro = decodeVector(frame, (2000.0f * DEG2RAD) / 32768.0f);
      haveGyro = true;
      break;
    case FRAME_ANGLE:
      angle = decodeVector(frame, 180.0f / 32768.0f);
      haveAngle = true;
      break;
    default:
      break;
  }
}

void printCsvSample() {
  Serial.print(acc.x, 6);
  Serial.print(",");
  Serial.print(acc.y, 6);
  Serial.print(",");
  Serial.print(acc.z, 6);
  Serial.print(",");
  Serial.print(gyro.x, 6);
  Serial.print(",");
  Serial.print(gyro.y, 6);
  Serial.print(",");
  Serial.println(gyro.z, 6);
}

void setup() {
  Serial.begin(DEBUG_BAUD);
  const unsigned long waitStartedMs = millis();
  while (!Serial && (millis() - waitStartedMs) < SERIAL_WAIT_TIMEOUT_MS) {
    delay(10);
  }

  Serial1.begin(SENSOR_BAUD);
  delay(200);

  Serial.println("INFO: WT61PC UART bridge started");
  Serial.println("INFO: Board: Arduino Nano 33 IoT");
  Serial.println("INFO: Expected sensor wiring: TXD->D0/RX1, RXD->D1/TX1, GND->GND");
  Serial.println("INFO: Output CSV: ax_mps2,ay_mps2,az_mps2,gx_rps,gy_rps,gz_rps");
}

void loop() {
  uint8_t frame[FRAME_LEN];
  while (readWt61pcFrame(frame)) {
    processFrame(frame);
    if (haveAcc && haveGyro && haveAngle) {
      printCsvSample();
      sampleCount++;
      haveAcc = false;
      haveGyro = false;
      haveAngle = false;
    }
  }

  const unsigned long nowMs = millis();
  if ((nowMs - lastStatusMs) >= STATUS_PERIOD_MS) {
    lastStatusMs = nowMs;
    if (sampleCount == 0) {
      Serial.println("WARN: no WT61PC frames yet; check TX/RX crossed, GND, VCC, and 9600 baud");
    }
  }
}
