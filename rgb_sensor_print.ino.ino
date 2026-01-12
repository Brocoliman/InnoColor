#include <Wire.h>
#include "Adafruit_TCS34725.h"

// Initialize sensor (default address 0x29)
Adafruit_TCS34725 tcs = Adafruit_TCS34725(TCS34725_INTEGRATIONTIME_50MS, TCS34725_GAIN_4X);

void setup() {
  Serial.begin(115200);
  if (!tcs.begin()) {
    Serial.println("[ERROR] Sensor not found!");
    while (1);
  }
  Serial.println("[EVENT] RGB Sensor ready!");
}

void loop() {
  uint16_t r, g, b, c; // store RGBA values
  
  tcs.getRawData(&r, &g, &b, &c); // read raw data

  // Normalize
  uint32_t sum = r + g + b;
  float red = (float)r / sum * 255;
  float green = (float)g / sum * 255;
  float blue = (float)b / sum * 255;

  Serial.print((int)red); Serial.print(",");
  Serial.print((int)green); Serial.print(",");
  Serial.println((int)blue);

  delay(1500);
}
