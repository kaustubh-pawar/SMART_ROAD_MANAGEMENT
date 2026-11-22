# Smart Road Management System

### ESP32-CAM + NEO-6M GPS + OpenCV + YOLO Road Defect Detection & Analytics Dashboard

An end-to-end intelligent road maintenance monitoring system that captures pavement images with an ESP32-CAM, geotags them with GPS coordinates from a NEO-6M module, uploads them via HTTP/HTTPS to a Flask backend, detects and classifies road defects (potholes, longitudinal cracks, transverse cracks, and alligator cracks) using computer vision and deep learning, computes a Road Condition Severity Index, and visualizes all data in an interactive dashboard with GPS mapping, manual upload analysis, and automated audit report generation.

---

## 1. System Architecture

```text
       [ Vehicle / Field Unit ]
Push Button  ──>  ESP32-CAM (SVGA)  ──>  NEO-6M GPS (Lat/Lng)
                      │
           (HTTP / HTTPS POST)
                      │
                      ▼
[ Flask Ingestion & Analytics Server ]
  ├── Save Original Image & Metadata JSON
  ├── OpenCV Preprocessing (CLAHE, Bilateral Filter, Black-Hat)
  ├── Defect Detection (YOLO / Advanced CV Engine)
  │     ├── Potholes (Red)
  │     ├── Longitudinal Cracks (Yellow)
  │     ├── Transverse Cracks (Cyan)
  │     └── Alligator Cracks (Orange)
  ├── Road Severity Rating (Good, Fair, Poor, Critical)
  ├── Save Annotated Image (`processed/`) & JSON (`detections/`)
  │
  ├── [ Interactive Web Dashboard ]
  │     ├── Live KPI Metric Cards
  │     ├── OpenStreetMap / Leaflet GPS Hotspots
  │     ├── Inspection Gallery (Original vs Processed)
  │     ├── Manual Image Drag & Drop Analyzer
  │     └── Daily, Weekly & Monthly Report Center (CSV/JSON Export)
  │
  └── [ Railway Cloud Deployment ]
        ├── Automatic HTTPS Domain
        └── Fleet / Multi-Device Scalability
```

---

## 2. Project Directory Structure

```text
SMART_ROAD_MANAGEMENT/
├── server.py                        # Unified Flask server, REST APIs & dashboard
├── process_image.py                 # OpenCV + YOLO defect detection engine
├── reports.py                       # Daily, Weekly & Monthly report generator (JSON & CSV)
├── train_model.py                   # Custom YOLO11/YOLOv8 training script for RDD2022
├── requirements.txt                 # Python dependencies
├── Procfile                         # Railway / Cloud WSGI runner
├── railway.json                     # Railway deployment configuration
├── Dockerfile                       # Production container definition
├── .dockerignore                    # Excluded files for Docker build
├── .gitignore                       # Git exclusion rules
├── templates/
│   └── index.html                   # Modern responsive Web Dashboard (Tailwind + Leaflet + Chart.js)
├── AI_MODELS/
│   └── pothole_crack.pt             # Trained YOLO model weights
├── CAM_001/
│   └── YYYY/
│       └── MM_MONTH/
│           └── DD_MON/
│               ├── original/        # Raw captured JPEGs
│               ├── metadata/        # GPS & device metadata JSONs
│               ├── processed/       # AI-annotated defect images
│               └── detections/      # Defect coordinates & severity JSONs
├── REPORTS/
│   ├── daily/                       # Daily inspection summaries (JSON & CSV)
│   ├── weekly/                      # Weekly inspection summaries (JSON & CSV)
│   └── monthly/                     # Monthly inspection summaries (JSON & CSV)
└── Smart_Road_ESP32_CAM.ino/
    └── Smart_Road_ESP32_CAM/
        └── Smart_Road_ESP32_CAM.ino # ESP32-CAM firmware (Cloud Railway & Local UDP)
```

---

## 3. Key Features

1. **Dual Defect Detection Engine (`process_image.py`)**:
   - **Custom YOLO**: Automatically loads trained weights from `AI_MODELS/pothole_crack.pt` when available.
   - **Computer Vision Fallback**: Uses OpenCV with CLAHE lighting normalization, morphological black-hat filtering, adaptive thresholding, and geometric contour analysis to detect and classify potholes, longitudinal cracks, transverse cracks, and alligator cracks.
   - **Severity Scoring**: Categorizes roads into `Good` (0 defects), `Fair` (minor fissures), `Poor` (moderate damage or single pothole), and `Critical` (multiple potholes or severe mesh cracks).

2. **Interactive Web Dashboard (`/`)**:
   - **KPI Cards**: Real-time totals for processed frames, defects, potholes, cracks, and critical zones.
   - **Defect Distribution Charts**: Interactive donut and bar charts (Chart.js) showing defect frequency and road condition distribution.
   - **GPS Defect Map**: Interactive Leaflet.js map with color-coded pulsing pins indicating severity.
   - **Inspection Gallery**: Grid view with original vs annotated image toggling and a deep inspection modal with exact bounding boxes `[x1, y1, x2, y2]`, confidence %, and pixel areas.
   - **Manual Image Upload**: Drag-and-drop or file selector to analyze any road image from phone or computer, with one-click device GPS coordinate acquisition.

3. **Report Generation System (`reports.py`)**:
   - Generates Daily, Weekly, and Monthly inspection reports.
   - Compiles hazard hotspots, defect density, road severity breakdowns, and camera activity.
   - Exports directly to audit-ready **CSV spreadsheets** and **JSON files** for municipal and road maintenance authorities.

4. **Railway Cloud Deployment & Multi-Device Support**:
   - Configured with `Procfile`, `railway.json`, and dynamic `$PORT` handling.
   - The ESP32-CAM firmware supports direct HTTPS uploads to Railway domains with `WiFiClientSecure.setInsecure()`.
   - Supports multi-device fleets: each camera can have its own ID (`CAM_001`, `CAM_002`, or automatic MAC-based ID `CAM_XXXX`).

---

## 4. Hardware Wiring & Pinout

### AI Thinker ESP32-CAM Connections

| Component | Pin | ESP32-CAM Pin | Notes |
|---|---|---|---|
| **NEO-6M GPS** | TX | GPIO 13 (U1RX) | Receives NMEA data from GPS |
| **NEO-6M GPS** | RX | GPIO 14 (U1TX) | Optional send pin |
| **NEO-6M GPS** | VCC | 3.3V / 5V | Power supply |
| **NEO-6M GPS** | GND | GND | Shared ground |
| **Push Button** | Pin 1 | GPIO 2 | Internal pull-up enabled |
| **Push Button** | Pin 2 | GND | Press pulls LOW to trigger capture |
| **FTDI (Flashing)** | TXD | GPIO 3 (U0R) | Serial receive |
| **FTDI (Flashing)** | RXD | GPIO 1 (U0T) | Serial transmit |
| **FTDI (Flashing)** | GND | GND | Ground |
| **FTDI (Flashing)** | 5V | 5V | Regulated 5V source |

> **Flashing Note:** Connect **GPIO 0 to GND** before powering on to flash firmware via FTDI. Disconnect GPIO 0 from GND and reset the board to run normally.

### Hardware Support & Connect Details
- **Team / Organization:** HYDL
- **Contact Number:** 9373529729
- **Assistance:** ESP32-CAM circuit setup, NEO-6M GPS integration, field testing, and troubleshooting.

---

## 5. Local Setup & Usage

### 1. Python Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python server.py
```
Open your browser at `http://localhost:5001/` to access the dashboard.

### 3. Batch Process Existing Images
To analyze all pending captured images:
```bash
python process_image.py --all
```

### 4. Generate Reports
To generate daily, weekly, and monthly reports via CLI:
```bash
python reports.py
```

### 5. Train Custom YOLO Model (RDD2022)
To train custom YOLO weights:
```bash
python train_model.py --epochs 50 --batch 8 --device auto
```

---

## 6. Railway Cloud Deployment Guide

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy Smart Road Management System"
   git push origin main
   ```

2. **Deploy on Railway**:
   - Go to [Railway.app](https://railway.app/).
   - Click **New Project** → **Deploy from GitHub repo** → Select this repository.
   - Railway will automatically detect the `Procfile` and `railway.json`.
   - In project settings, click **Generate Domain** to get your public HTTPS URL (e.g. `https://smart-road-production.up.railway.app`).

3. **Configure ESP32-CAM Firmware**:
   - In `Smart_Road_ESP32_CAM.ino`, set:
     ```cpp
     const bool USE_CLOUD_SERVER = true;
     const char* CLOUD_SERVER_URL = "https://your-project.up.railway.app/upload";
     const char* CONFIGURED_CAMERA_ID = "CAM_001"; // or "AUTO" for multi-device
     ```
   - Flash the ESP32-CAM. The hardware will now upload directly to your deployed Railway cloud project from any mobile Wi-Fi hotspot or cellular router.
