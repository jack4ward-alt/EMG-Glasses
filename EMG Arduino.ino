/*
  EMG Sub-Vocal Recorder — Arduino Sketch
  ========================================
  Reads 4 MyoWare 2.0 sensors on analog pins A0-A3
  and sends readings over serial to the Python recorder.

  WIRING:
    Sensor 1 (jaw left)    → VIN=5V, GND=GND, ENV=A0
    Sensor 2 (jaw right)   → VIN=5V, GND=GND, ENV=A1
    Sensor 3 (throat left) → VIN=5V, GND=GND, ENV=A2
    Sensor 4 (throat right)→ VIN=5V, GND=GND, ENV=A3

  OUTPUT FORMAT (over serial):
    val0,val1,val2,val3\n
    e.g. 512,487,601,390

  Each value is 0-1023 (10-bit ADC reading)
*/

const int SENSOR_PINS[4] = {A0, A1, A2, A3};
const int NUM_SENSORS     = 4;
const int SAMPLE_DELAY_US = 2000;  // 2ms = 500Hz sample rate

void setup() {
  Serial.begin(115200);
  // Set analog reference to default (5V on Uno)
  analogReference(DEFAULT);
}

void loop() {
  // Read all 4 sensors
  for (int i = 0; i < NUM_SENSORS; i++) {
    int val = analogRead(SENSOR_PINS[i]);
    Serial.print(val);
    if (i < NUM_SENSORS - 1) {
      Serial.print(",");
    }
  }
  Serial.println();  // Newline ends the sample

  // Wait to hit target sample rate
  delayMicroseconds(SAMPLE_DELAY_US);
}
