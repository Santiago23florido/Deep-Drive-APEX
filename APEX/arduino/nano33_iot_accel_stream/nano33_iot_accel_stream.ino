#include <Arduino_LSM6DS3.h>

/*
  APEX - Nano 33 IoT IMU stream

  This sketch reads 3-axis acceleration and 3-axis angular velocity
  from the onboard IMU and streams them over USB serial in CSV format:
      ax,ay,az,gx,gy,gz

  Acceleration units: m/s^2
  Angular velocity units: rad/s

  This format is consumed by the Raspberry APEX serial node.
  Backward note:
  - Older APEX versions only consumed ax,ay,az.
  - Current APEX consumes all six values and uses gyro for yaw/odometry.
*/

static constexpr float GRAVITY_MPS2 = 9.80665f;
static constexpr float DEG2RAD = 0.017453292519943295f;
static constexpr unsigned long STREAM_PERIOD_MS = 10;  // ~100 Hz
static constexpr unsigned long SERIAL_WAIT_TIMEOUT_MS = 2500;

void setup() {
  Serial.begin(115200);
  const unsigned long serial_wait_started_ms = millis();
  while (!Serial && (millis() - serial_wait_started_ms) < SERIAL_WAIT_TIMEOUT_MS) {
    delay(10);
  }

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("INFO: Nano IMU stream started (ax,ay,az,gx,gy,gz)");
}

void loop() {
  float ax_g = 0.0f;
  float ay_g = 0.0f;
  float az_g = 0.0f;
  float gx_dps = 0.0f;
  float gy_dps = 0.0f;
  float gz_dps = 0.0f;

  // Publish one line only when both sensors have fresh samples.
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    IMU.readAcceleration(ax_g, ay_g, az_g);
    IMU.readGyroscope(gx_dps, gy_dps, gz_dps);

    const float ax_mps2 = ax_g * GRAVITY_MPS2;
    const float ay_mps2 = ay_g * GRAVITY_MPS2;
    const float az_mps2 = az_g * GRAVITY_MPS2;
    const float gx_rps = gx_dps * DEG2RAD;
    const float gy_rps = gy_dps * DEG2RAD;
    const float gz_rps = gz_dps * DEG2RAD;

    Serial.print(ax_mps2, 6);
    Serial.print(",");
    Serial.print(ay_mps2, 6);
    Serial.print(",");
    Serial.print(az_mps2, 6);
    Serial.print(",");
    Serial.print(gx_rps, 6);
    Serial.print(",");
    Serial.print(gy_rps, 6);
    Serial.print(",");
    Serial.println(gz_rps, 6);
  }

  delay(STREAM_PERIOD_MS);
}
