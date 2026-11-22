"""
Smart Road Management System - Unified Server & API Gateway
Provides ESP32-CAM ingestion endpoint, manual dashboard upload, live OpenCV/YOLO
defect detection, report generation, and interactive web dashboard.
Fully compatible with local execution and Railway cloud deployment.
"""

import os
import sys
import json
import re
import socket
import threading
import importlib
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, abort

# Import detection and reporting modules
import process_image
import reports

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

# Configuration: support dynamic PORT from Railway/Render/Heroku or fallback to 5001 locally
HTTP_PORT = int(os.environ.get("PORT", 5001))
DISCOVERY_PORT = int(os.environ.get("DISCOVERY_PORT", 4210))
ENABLE_UDP = os.environ.get("ENABLE_UDP", "true").lower() == "true"

# Maximum file upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


def safe_name(value, default="UNKNOWN"):
    """Sanitize directory and file names."""
    value = str(value or default)
    return re.sub(r"[^A-Za-z0-9_-]", "_", value)


def parse_timestamp(value):
    """Parse ISO timestamp or fallback to current datetime."""
    try:
        if value and value != "TIME_NOT_SYNCED":
            return datetime.fromisoformat(value)
    except Exception:
        pass
    return datetime.now()


# ----------------------------------------------------------------------
# WEB DASHBOARD & HEALTH ROUTES
# ----------------------------------------------------------------------

@app.route("/", methods=["GET"])
def dashboard():
    """Serve the Smart Road Management Web Dashboard."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def healthcheck():
    """Health check endpoint for Railway and cloud monitoring."""
    return jsonify({
        "status": "healthy",
        "service": "Smart Road Management System",
        "timestamp": datetime.now().isoformat(),
        "environment": "railway" if os.environ.get("RAILWAY_ENVIRONMENT") else "local"
    }), 200


@app.route("/api/config", methods=["GET"])
def get_config():
    """Return public URLs and connection configuration."""
    host_header = request.host
    scheme = "https" if request.is_secure or request.headers.get("X-Forwarded-Proto") == "https" else "http"
    base_url = f"{scheme}://{host_header}"

    return jsonify({
        "base_url": base_url,
        "upload_endpoint": f"{base_url}/upload",
        "manual_upload_endpoint": f"{base_url}/api/manual-upload",
        "http_port": HTTP_PORT,
        "discovery_port": DISCOVERY_PORT,
        "is_cloud": bool(os.environ.get("RAILWAY_ENVIRONMENT") or "railway" in host_header)
    })


# ----------------------------------------------------------------------
# HARDWARE INGESTION: ESP32-CAM / IOT DEVICES
# ----------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Hardware endpoint: Receives raw JPEG bytes from ESP32-CAM or external devices.
    Headers:
      - X-Camera-ID: device identifier (default CAM_001)
      - X-Latitude: GPS latitude
      - X-Longitude: GPS longitude
      - X-GPS-Valid: true / false
      - X-Timestamp: capture timestamp (ISO)
    Saves image & metadata, runs OpenCV/YOLO defect detection immediately.
    """
    image_data = request.get_data()

    if not image_data:
        return jsonify({"error": "No image data received"}), 400

    # Basic JPEG signature check
    if not image_data.startswith(b"\xff\xd8"):
        return jsonify({"error": "Payload is not a valid JPEG"}), 400

    camera_id = safe_name(request.headers.get("X-Camera-ID"), "CAM_001")
    latitude = request.headers.get("X-Latitude", "NA")
    longitude = request.headers.get("X-Longitude", "NA")
    gps_valid = request.headers.get("X-GPS-Valid", "false").lower() == "true"
    timestamp_text = request.headers.get("X-Timestamp", "")
    capture_time = parse_timestamp(timestamp_text)

    # Date-organized directory structure per README
    # <camera_id>/<YYYY>/<MM_MONTH>/<DD_MON>/
    year_folder = capture_time.strftime("%Y")
    month_folder = capture_time.strftime("%m_%B").upper()
    day_folder = capture_time.strftime("%d_%b").upper()

    day_path = BASE_DIR / camera_id / year_folder / month_folder / day_folder
    original_dir = day_path / "original"
    metadata_dir = day_path / "metadata"

    original_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    file_timestamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"{camera_id}_{file_timestamp}.jpg"
    image_path = original_dir / image_filename

    # Save original JPEG
    image_path.write_bytes(image_data)

    # Save metadata JSON
    metadata = {
        "camera_id": camera_id,
        "image_filename": image_filename,
        "capture_timestamp": timestamp_text or capture_time.isoformat(timespec="seconds"),
        "server_received_timestamp": datetime.now().isoformat(timespec="seconds"),
        "latitude": latitude,
        "longitude": longitude,
        "gps_valid": gps_valid,
        "image_size_bytes": len(image_data),
        "image_path": str(image_path.relative_to(BASE_DIR)),
        "processing_status": "pending"
    }

    metadata_filename = image_path.stem + ".json"
    metadata_path = metadata_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    # Run defect detection and annotation immediately
    detection_result = {}
    try:
        importlib.reload(process_image)
        detection_result = process_image.process_single_image(image_path, metadata_path)
    except Exception as e:
        print(f"[Upload Error] Detection failed on {image_filename}: {e}")

    print("\n--------------------------------")
    print(f"IMAGE RECEIVED & PROCESSED: {image_filename}")
    print(f"Camera ID: {camera_id} | GPS: {latitude}, {longitude} (valid: {gps_valid})")
    print(f"Defects: {detection_result.get('defect_count', 0)} | Condition: {detection_result.get('severity', 'Good')}")
    print("--------------------------------\n")

    return jsonify({
        "status": "success",
        "message": "Image received, stored, and analyzed",
        "camera_id": camera_id,
        "filename": image_filename,
        "gps_valid": gps_valid,
        "defect_count": detection_result.get("defect_count", 0),
        "severity": detection_result.get("severity", "Good"),
        "severity_score": detection_result.get("severity_score", 0.0),
        "processed_image_path": detection_result.get("processed_image_path", "")
    }), 200


# ----------------------------------------------------------------------
# DASHBOARD MANUAL IMAGE UPLOAD
# ----------------------------------------------------------------------

@app.route("/api/manual-upload", methods=["POST"])
def manual_upload():
    """
    Dashboard manual upload endpoint.
    Accepts multipart/form-data:
      - image: File
      - camera_id: optional string (defaults to MANUAL_WEB)
      - latitude: optional float/string
      - longitude: optional float/string
      - gps_valid: optional boolean
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    image_data = file.read()
    if not image_data.startswith(b"\xff\xd8") and not image_data.startswith(b"\x89PNG"):
        # Check basic JPEG/PNG signature
        return jsonify({"error": "Uploaded file is not a supported JPEG or PNG image"}), 400

    camera_id = safe_name(request.form.get("camera_id"), "MANUAL_WEB")
    latitude = request.form.get("latitude", "NA").strip() or "NA"
    longitude = request.form.get("longitude", "NA").strip() or "NA"
    gps_valid = request.form.get("gps_valid", "false").lower() == "true" or (latitude != "NA" and longitude != "NA")

    capture_time = datetime.now()
    year_folder = capture_time.strftime("%Y")
    month_folder = capture_time.strftime("%m_%B").upper()
    day_folder = capture_time.strftime("%d_%b").upper()

    day_path = BASE_DIR / camera_id / year_folder / month_folder / day_folder
    original_dir = day_path / "original"
    metadata_dir = day_path / "metadata"

    original_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    file_timestamp = capture_time.strftime("%Y%m%d_%H%M%S_%f")
    # Clean extension
    ext = ".jpg" if image_data.startswith(b"\xff\xd8") else ".png"
    image_filename = f"{camera_id}_{file_timestamp}{ext}"
    image_path = original_dir / image_filename

    image_path.write_bytes(image_data)

    metadata = {
        "camera_id": camera_id,
        "image_filename": image_filename,
        "capture_timestamp": capture_time.isoformat(timespec="seconds"),
        "server_received_timestamp": capture_time.isoformat(timespec="seconds"),
        "latitude": latitude,
        "longitude": longitude,
        "gps_valid": gps_valid,
        "image_size_bytes": len(image_data),
        "image_path": str(image_path.relative_to(BASE_DIR)),
        "processing_status": "pending",
        "source": "manual_dashboard_upload"
    }

    metadata_filename = image_path.stem + ".json"
    metadata_path = metadata_dir / metadata_filename
    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")

    # Run detection
    try:
        importlib.reload(process_image)
        detection_result = process_image.process_single_image(image_path, metadata_path)
    except Exception as e:
        return jsonify({"error": f"Defect analysis failed: {str(e)}"}), 500

    return jsonify(detection_result), 200


# ----------------------------------------------------------------------
# DASHBOARD DATA & ANALYTICS APIS
# ----------------------------------------------------------------------

@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Calculate and return system KPI summary metrics for the dashboard."""
    records = reports.get_all_detection_records()
    total_captures = len(records)
    total_defects = 0

    defects_by_class = {
        "pothole": 0,
        "longitudinal_crack": 0,
        "transverse_crack": 0,
        "alligator_crack": 0
    }
    severity_breakdown = {
        "Good": 0,
        "Fair": 0,
        "Poor": 0,
        "Critical": 0
    }
    cameras = set()
    gps_count = 0

    for r in records:
        total_defects += r.get("defect_count", 0)
        sev = r.get("severity", "Good")
        severity_breakdown[sev] = severity_breakdown.get(sev, 0) + 1
        cam_id = r.get("camera_id")
        if cam_id:
            cameras.add(cam_id)
        if r.get("gps_valid") and r.get("latitude") != "NA":
            gps_count += 1

        summary = r.get("summary", {})
        for k, v in summary.items():
            if k in defects_by_class:
                defects_by_class[k] += v

    density = round(total_defects / total_captures, 2) if total_captures > 0 else 0.0

    return jsonify({
        "total_captures": total_captures,
        "total_defects": total_defects,
        "defect_density": density,
        "defects_by_class": defects_by_class,
        "severity_breakdown": severity_breakdown,
        "gps_tagged_count": gps_count,
        "active_cameras": list(cameras),
        "last_updated": datetime.now().isoformat()
    })


@app.route("/api/records", methods=["GET"])
def get_records():
    """Return all detection records with optional filters."""
    records = reports.get_all_detection_records()

    camera_id = request.args.get("camera_id")
    severity = request.args.get("severity")

    if camera_id and camera_id != "ALL":
        records = [r for r in records if r.get("camera_id") == camera_id]
    if severity and severity != "ALL":
        records = [r for r in records if r.get("severity") == severity]

    return jsonify({
        "count": len(records),
        "records": records
    })


@app.route("/api/record/<image_filename>", methods=["GET"])
def get_single_record(image_filename):
    """Retrieve complete metadata and detection details for a specific image."""
    records = reports.get_all_detection_records()
    for r in records:
        if r.get("image_filename") == image_filename:
            return jsonify(r)
    return jsonify({"error": "Record not found"}), 404


@app.route("/api/reprocess/<image_filename>", methods=["POST"])
def reprocess_record(image_filename):
    """Re-run OpenCV/YOLO detection on a specific captured image."""
    base_images = list(BASE_DIR.glob(f"**/original/{image_filename}"))
    if not base_images:
        return jsonify({"error": "Image file not found"}), 404

    target_img = base_images[0]
    try:
        importlib.reload(process_image)
        result = process_image.process_single_image(target_img)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Reprocessing failed: {str(e)}"}), 500


@app.route("/api/delete/<image_filename>", methods=["DELETE", "POST"])
def delete_record(image_filename):
    """
    Delete a capture and all associated files:
    - original image
    - processed image
    - metadata JSON
    - detection JSON
    """
    stem = Path(image_filename).stem
    deleted_files = []

    # Find and delete matching files across original, processed, metadata, detections
    patterns = [
        f"**/original/{image_filename}",
        f"**/processed/{image_filename}",
        f"**/metadata/{stem}.json",
        f"**/detections/{stem}.json"
    ]

    for pat in patterns:
        for f in BASE_DIR.glob(pat):
            try:
                if f.is_file():
                    f.unlink()
                    deleted_files.append(str(f.relative_to(BASE_DIR)))
            except Exception as e:
                print(f"[Delete Error] Could not delete {f}: {e}")

    if not deleted_files:
        return jsonify({"error": "No matching files found to delete"}), 404

    # Automatically refresh reports after deletion
    try:
        importlib.reload(reports)
        reports.generate_daily_report()
        reports.generate_weekly_report()
        reports.generate_monthly_report()
    except Exception as e:
        print(f"[Warning] Failed to regenerate reports after deletion: {e}")

    return jsonify({
        "status": "success",
        "message": f"Successfully deleted {len(deleted_files)} file(s) for {image_filename}",
        "deleted_files": deleted_files
    }), 200


@app.route("/api/process-all", methods=["POST"])
def process_all_unprocessed():
    """Batch process all captured images that have not been processed yet."""
    try:
        importlib.reload(process_image)
        results = process_image.batch_process_all_pending(BASE_DIR)
        return jsonify({
            "status": "success",
            "processed_count": len(results),
            "message": f"Successfully processed {len(results)} image(s)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
# REPORT GENERATION & DOWNLOADS
# ----------------------------------------------------------------------

@app.route("/api/reports/list", methods=["GET"])
def get_reports_list():
    """List all archived reports."""
    return jsonify(reports.list_saved_reports())


@app.route("/api/reports/generate", methods=["POST"])
def trigger_report():
    """Generate a daily, weekly, or monthly report."""
    data = request.get_json() or {}
    report_type = data.get("type", "daily").lower()

    importlib.reload(reports)
    if report_type == "daily":
        rep = reports.generate_daily_report()
    elif report_type == "weekly":
        rep = reports.generate_weekly_report()
    elif report_type == "monthly":
        rep = reports.generate_monthly_report()
    else:
        return jsonify({"error": "Invalid report type (choose daily, weekly, monthly)"}), 400

    return jsonify(rep)


@app.route("/api/reports/download/<file_type>/<filename>", methods=["GET"])
def download_report(file_type, filename):
    """Download report JSON or CSV file."""
    # Find matching report file inside REPORTS
    candidates = list(reports.REPORTS_DIR.glob(f"**/{filename}"))
    if not candidates:
        return abort(404, description="Report file not found")

    target = candidates[0]
    return send_from_directory(
        directory=str(target.parent),
        path=target.name,
        as_attachment=True
    )


# ----------------------------------------------------------------------
# STATIC FILE SERVING FOR CAPTURED & PROCESSED IMAGES
# ----------------------------------------------------------------------

@app.route("/files/<path:filepath>")
def serve_file(filepath):
    """Safely serve original or processed images stored in the project."""
    target_path = (BASE_DIR / filepath).resolve()
    # Security check: must reside inside BASE_DIR
    if not str(target_path).startswith(str(BASE_DIR)):
        return abort(403)
    if not target_path.exists() or not target_path.is_file():
        return abort(404)

    return send_from_directory(str(target_path.parent), target_path.name)


# ----------------------------------------------------------------------
# UDP LOCAL DISCOVERY SERVICE (BACKGROUND THREAD)
# ESP32-CAM broadcasts locally on port 4210 to discover this server.
# Gracefully disabled or skipped in restricted cloud environments.
# ----------------------------------------------------------------------

def run_udp_discovery():
    """Listen for local ESP32-CAM broadcast discovery requests."""
    try:
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind(("0.0.0.0", DISCOVERY_PORT))
        print(f"[Discovery] UDP discovery service active on port {DISCOVERY_PORT}")

        while True:
            data, client_address = udp.recvfrom(1024)
            message = data.decode("utf-8", errors="ignore").strip()

            if message != "DISCOVER_SMART_ROAD_SERVER":
                continue

            # Determine interface IP communicating with ESP32-CAM
            route_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                route_socket.connect((client_address[0], client_address[1]))
                laptop_ip = route_socket.getsockname()[0]
            except Exception:
                laptop_ip = "127.0.0.1"
            finally:
                route_socket.close()

            server_url = f"http://{laptop_ip}:{HTTP_PORT}/upload"
            udp.sendto(server_url.encode("utf-8"), client_address)
            print(f"[Discovery] ESP32-CAM discovered at {client_address[0]} -> Reply: {server_url}")

    except Exception as e:
        print(f"[Discovery] UDP discovery unavailable or restricted in this environment: {e}")


# ----------------------------------------------------------------------
# START SERVER
# ----------------------------------------------------------------------

if __name__ == "__main__":
    if ENABLE_UDP:
        disc_thread = threading.Thread(target=run_udp_discovery, daemon=True)
        disc_thread.start()

    print("\n=======================================================")
    print("      SMART ROAD MANAGEMENT SERVER RUNNING             ")
    print("=======================================================")
    print(f" Web Dashboard:       http://localhost:{HTTP_PORT}/")
    print(f" Hardware Endpoint:   http://localhost:{HTTP_PORT}/upload")
    print(f" Manual Upload API:   http://localhost:{HTTP_PORT}/api/manual-upload")
    print(f" UDP Discovery Port:  {DISCOVERY_PORT}")
    print(f" Working Directory:   {BASE_DIR}")
    print("=======================================================\n")

    app.run(host="0.0.0.0", port=HTTP_PORT, debug=False)
