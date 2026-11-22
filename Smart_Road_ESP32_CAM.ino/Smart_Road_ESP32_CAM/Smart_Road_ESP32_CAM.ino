#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <WiFiClientSecure.h>
#include <TinyGPSPlus.h>
#include <time.h>

// ==================================================
// WIFI CREDENTIALS
// ==================================================

const char* WIFI_SSID = "KP";
const char* WIFI_PASSWORD = "10060711";

// ==================================================
// SERVER & DEPLOYMENT CONFIGURATION
// ==================================================

// Set to 'true' to connect directly to your Railway deployed cloud project
// Set to 'false' to use local network UDP auto-discovery with your laptop
const bool USE_CLOUD_SERVER = true;

// Your deployed Railway project endpoint (e.g., https://smart-road-production.up.railway.app/upload)
// Replace with your actual Railway URL after deploying
const char* CLOUD_SERVER_URL = "https://your-project.up.railway.app/upload";

// Local server fallback URL (if discovery is bypassed)
String serverURL = "";

// ==================================================
// CAMERA AND MULTI-DEVICE SETTINGS
// ==================================================

// Unique Camera ID for this unit (e.g., "CAM_001", "CAM_002")
// Set to "AUTO" to automatically generate an ID from the ESP32 MAC address
const char* CONFIGURED_CAMERA_ID = "CAM_001";

#define BUTTON_PIN 2

// GPS wiring:
// GPS TX -> ESP32 GPIO 13 (ESP32 UART RX)
// GPS RX -> ESP32 GPIO 14 (ESP32 UART TX - optional)
#define GPS_RX_PIN 13
#define GPS_TX_PIN 14

// Onboard flash LED pin (optional feedback on AI Thinker)
#define FLASH_LED_PIN 4

// ==================================================
// DISCOVERY SETTINGS (LOCAL NETWORK)
// ==================================================

WiFiUDP discoveryUDP;
const int DISCOVERY_PORT = 4210;
const int ESP32_UDP_PORT = 4211;

// ==================================================
// GPS INSTANCE
// ==================================================

TinyGPSPlus gps;
HardwareSerial GPS_Serial(1);

// ==================================================
// AI THINKER ESP32-CAM PIN DEFINITIONS
// ==================================================

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0

#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// ==================================================
// BUTTON DEBOUNCE STATE
// ==================================================

bool lastButtonReading = HIGH;
bool stableButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Reconnection timer
unsigned long lastWiFiCheck = 0;
const unsigned long wifiCheckInterval = 10000;

// ==================================================
// HELPER: GET UNIQUE CAMERA ID
// ==================================================

String getCameraID() {
  if (String(CONFIGURED_CAMERA_ID) != "AUTO" && strlen(CONFIGURED_CAMERA_ID) > 0) {
    return String(CONFIGURED_CAMERA_ID);
  }

  // Derive unique ID from chip MAC address (e.g. CAM_A1B2C3)
  uint8_t mac[6];
  WiFi.macAddress(mac);
  char autoId[20];
  snprintf(autoId, sizeof(autoId), "CAM_%02X%02X%02X", mac[3], mac[4], mac[5]);
  return String(autoId);
}

// ==================================================
// HELPER: BLINK FLASH LED FOR FEEDBACK
// ==================================================

void signalFeedback(int count, int delayMs) {
  pinMode(FLASH_LED_PIN, OUTPUT);
  for (int i = 0; i < count; i++) {
    digitalWrite(FLASH_LED_PIN, HIGH);
    delay(delayMs);
    digitalWrite(FLASH_LED_PIN, LOW);
    if (i < count - 1) delay(delayMs);
  }
}

// ==================================================
// GPS UPDATE
// ==================================================

void updateGPS() {
  while (GPS_Serial.available() > 0) {
    gps.encode(GPS_Serial.read());
  }
}

// ==================================================
// GET ISO TIMESTAMP
// ==================================================

String getTimestamp() {
  struct tm timeinfo;
  if (getLocalTime(&timeinfo, 1000)) {
    char buffer[30];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S", &timeinfo);
    return String(buffer);
  }
  return "TIME_NOT_SYNCED";
}

// ==================================================
// CAMERA INITIALIZATION
// ==================================================

bool startCamera() {
  camera_config_t config = {};

  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;

  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;

  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_SVGA; // 800x600 for sharp road detail
    config.jpeg_quality = 10;
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_VGA;  // 640x480 fallback
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  esp_err_t result = esp_camera_init(&config);
  if (result != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", result);
    return false;
  }

  Serial.println("Camera initialized successfully.");
  return true;
}

// ==================================================
// WIFI CONNECTION & AUTO-RECONNECT
// ==================================================

bool connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }

  Serial.println("Connecting to WiFi: " + String(WIFI_SSID) + "...");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("Device IP: ");
    Serial.println(WiFi.localIP());
    return true;
  }

  Serial.println("\nWiFi connection failed. Retrying when button is pressed.");
  return false;
}

// ==================================================
// LOCAL LAPTOP SERVER DISCOVERY (UDP)
// ==================================================

bool discoverServer() {
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  Serial.println("Broadcasting UDP discovery on local network...");
  IPAddress broadcastIP(255, 255, 255, 255);

  for (int attempt = 0; attempt < 5; attempt++) {
    discoveryUDP.beginPacket(broadcastIP, DISCOVERY_PORT);
    discoveryUDP.print("DISCOVER_SMART_ROAD_SERVER");
    discoveryUDP.endPacket();

    unsigned long startTime = millis();
    while (millis() - startTime < 1000) {
      updateGPS();
      int packetSize = discoveryUDP.parsePacket();
      if (packetSize > 0) {
        String response = discoveryUDP.readString();
        response.trim();

        if (response.startsWith("http://") && response.endsWith("/upload")) {
          serverURL = response;
          Serial.println("Local server discovered: " + serverURL);
          return true;
        }
      }
      delay(10);
    }
  }

  Serial.println("Local server not found via UDP.");
  return false;
}

// ==================================================
// CAPTURE AND UPLOAD TO CLOUD OR LOCAL SERVER
// ==================================================

void captureAndUpload() {
  if (!connectWiFi()) {
    Serial.println("Upload cancelled: WiFi unavailable.");
    return;
  }

  // Determine active target endpoint
  String targetEndpoint = "";
  if (USE_CLOUD_SERVER) {
    targetEndpoint = String(CLOUD_SERVER_URL);
  } else {
    if (serverURL == "") {
      if (!discoverServer()) {
        Serial.println("Upload cancelled: Local server not discovered.");
        return;
      }
    }
    targetEndpoint = serverURL;
  }

  // Keep GPS data updated
  updateGPS();

  Serial.println("\nCapturing road image...");
  camera_fb_t* fb = esp_camera_fb_get();
  if (fb == NULL) {
    Serial.println("Camera capture failed!");
    return;
  }

  // Read GPS coordinates
  String latitude = "NA";
  String longitude = "NA";
  String gpsValid = "false";

  if (gps.location.isValid() && gps.location.age() < 5000) {
    latitude = String(gps.location.lat(), 6);
    longitude = String(gps.location.lng(), 6);
    gpsValid = "true";
  }

  String timestamp = getTimestamp();
  String camId = getCameraID();

  Serial.println("----------------------------------------");
  Serial.println("Frame Captured");
  Serial.println("Device ID: " + camId);
  Serial.printf("Size:      %u bytes\n", fb->len);
  Serial.println("Latitude:  " + latitude);
  Serial.println("Longitude: " + longitude);
  Serial.println("GPS Valid: " + gpsValid);
  Serial.println("Timestamp: " + timestamp);
  Serial.println("Target:    " + targetEndpoint);
  Serial.println("----------------------------------------");

  // HTTP / HTTPS Client
  HTTPClient http;
  WiFiClientSecure secureClient;
  WiFiClient regularClient;

  bool isHttps = targetEndpoint.startsWith("https://");

  if (isHttps) {
    // Crucial for Railway HTTPS: allows secure TLS without hardcoding certificates
    secureClient.setInsecure();
    http.begin(secureClient, targetEndpoint);
  } else {
    http.begin(regularClient, targetEndpoint);
  }

  http.setTimeout(25000); // 25s timeout for mobile uploads
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-Camera-ID", camId);
  http.addHeader("X-Latitude", latitude);
  http.addHeader("X-Longitude", longitude);
  http.addHeader("X-GPS-Valid", gpsValid);
  http.addHeader("X-Timestamp", timestamp);

  Serial.println("Uploading image payload to server...");
  int responseCode = http.POST(fb->buf, fb->len);

  if (responseCode > 0) {
    Serial.printf("HTTP Response code: %d\n", responseCode);
    String response = http.getString();
    Serial.println("Server Response:\n" + response);

    if (responseCode == 200) {
      Serial.println("SUCCESS: Frame processed and stored by server.");
      signalFeedback(2, 60); // Quick double flash
    }
  } else {
    Serial.print("Upload Error: ");
    Serial.println(http.errorToString(responseCode));

    if (!USE_CLOUD_SERVER) {
      // Reset local server URL so discovery runs again on next press
      serverURL = "";
    }
  }

  http.end();
  esp_camera_fb_return(fb);
  Serial.println("Ready for next capture.\n");
}

// ==================================================
// SETUP
// ==================================================

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n================================================");
  Serial.println("  SMART ROAD MANAGEMENT - ESP32-CAM NODE       ");
  Serial.println("================================================");

  pinMode(BUTTON_PIN, INPUT_PULLUP);

  // Initialize GPS UART on GPIO 13 (RX) and GPIO 14 (TX)
  GPS_Serial.begin(9600, SERIAL_8N1, GPS_RX_PIN, GPS_TX_PIN);

  if (!startCamera()) {
    Serial.println("FATAL: Camera initialization failed. Check ribbon cable.");
    while (true) {
      delay(1000);
    }
  }

  if (connectWiFi()) {
    // Configure NTP time (India Standard Time: UTC +5:30)
    configTime(19800, 0, "pool.ntp.org", "time.google.com");

    if (!USE_CLOUD_SERVER) {
      discoveryUDP.begin(ESP32_UDP_PORT);
      discoverServer();
    } else {
      Serial.println("Cloud Mode Active: " + String(CLOUD_SERVER_URL));
    }
  }

  Serial.println("\nSmart Road Management Node initialized.");
  Serial.println("Camera ID: " + getCameraID());
  Serial.println("Press the push button (GPIO 2) to capture and inspect road.\n");
}

// ==================================================
// MAIN LOOP
// ==================================================

void loop() {
  // Continuously feed GPS NMEA sentences
  updateGPS();

  // Periodically check Wi-Fi health
  if (millis() - lastWiFiCheck > wifiCheckInterval) {
    lastWiFiCheck = millis();
    if (WiFi.status() != WL_CONNECTED) {
      connectWiFi();
    }
  }

  // Push button debounce logic
  bool reading = digitalRead(BUTTON_PIN);
  if (reading != lastButtonReading) {
    lastDebounceTime = millis();
  }

  if (millis() - lastDebounceTime > debounceDelay) {
    if (reading != stableButtonState) {
      stableButtonState = reading;
      // Button pressed (GPIO 2 grounded)
      if (stableButtonState == LOW) {
        captureAndUpload();
      }
    }
  }

  lastButtonReading = reading;
  delay(5);
}
