#include <Arduino_LSM6DS3.h>

/*
  APEX - Nano 33 IoT accelerometer stream

  This sketch reads the 3-axis acceleration from the onboard IMU
  and streams it over USB serial in CSV format:
      ax,ay,az

  Units are m/s^2 to simplify processing on the Raspberry side.
*/

static constexpr float GRAVITY_MPS2 = 9.80665f;
static constexpr unsigned long STREAM_PERIOD_MS = 10;  // ~100 Hz

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ;  // Wait for USB serial on boards that need it.
  }

  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed");
    while (true) {
      delay(1000);
    }
  }

  Serial.println("INFO: Nano IMU stream started (m/s^2)");
}

void loop() {
  float ax_g = 0.0f;
  float ay_g = 0.0f;
  float az_g = 0.0f;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax_g, ay_g, az_g);

    const float ax_mps2 = ax_g * GRAVITY_MPS2;
    const float ay_mps2 = ay_g * GRAVITY_MPS2;
    const float az_mps2 = az_g * GRAVITY_MPS2;

    Serial.print(ax_mps2, 6);
    Serial.print(",");
    Serial.print(ay_mps2, 6);
    Serial.print(",");
    Serial.println(az_mps2, 6);
  }

  delay(STREAM_PERIOD_MS);
}
