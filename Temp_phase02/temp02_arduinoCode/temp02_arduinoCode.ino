#include <ESP8266WiFi.h>
#include <Wire.h>
#include <MPU6050.h>

const char* ssid = "Shriyash";
const char* password = "11112222";

const char* serverIP = "10.30.211.146"; // YOUR LAPTOP IP
const int serverPort = 5001;

WiFiClient client;
MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  mpu.initialize();

  if (!mpu.testConnection()) {
    Serial.println("MPU6050 connection failed");
  } else {
    Serial.println("MPU6050 connected");
  }

  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi Connected");
  Serial.print("ESP IP: ");
  Serial.println(WiFi.localIP());
}

void loop() {

  // Try connecting to server
  if (!client.connected()) {
    Serial.println("Connecting to server...");
    if (!client.connect(serverIP, serverPort)) {
      Serial.println("Connection failed");
      delay(1000);
      return;
    }
    Serial.println("Connected to server");
  }

  int16_t ax, ay, az, gx, gy, gz;
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);

  float ax_f = ax / 16384.0;
  float ay_f = ay / 16384.0;
  float az_f = az / 16384.0;

  float gx_f = gx / 131.0;
  float gy_f = gy / 131.0;
  float gz_f = gz / 131.0;

  String data = String(ax_f) + "," + String(ay_f) + "," + String(az_f) + "," +
                String(gx_f) + "," + String(gy_f) + "," + String(gz_f) + "\n";

  client.print(data);

  delay(40); // same as before (~25 samples/sec)
}