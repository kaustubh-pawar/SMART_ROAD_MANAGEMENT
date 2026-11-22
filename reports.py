"""
Smart Road Management System - Report Generation Module
Generates Daily, Weekly, and Monthly road inspection reports,
aggregates statistics, identifies critical defect hotspots, and exports to JSON and CSV.
"""

import os
import sys
import json
import csv
from pathlib import Path
from datetime import datetime, date, timedelta
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "REPORTS"


def get_all_detection_records(base_dir: Path = BASE_DIR) -> list:
    """
    Search and load all detection JSON records across all cameras and dates.
    """
    base_dir = Path(base_dir)
    detection_files = list(base_dir.glob("**/detections/*.json"))
    records = []

    for df in detection_files:
        try:
            data = json.loads(df.read_text(encoding="utf-8"))
            data["_file_path"] = str(df.relative_to(base_dir))
            records.append(data)
        except Exception as e:
            print(f"[Report] Error reading {df}: {e}")

    # Sort chronologically by capture or processed timestamp
    records.sort(
        key=lambda r: r.get("capture_timestamp") or r.get("processed_timestamp") or "",
        reverse=True
    )
    return records


def parse_record_date(record: dict) -> date:
    """Extract date from capture_timestamp, processed_timestamp, or file path."""
    for key in ["capture_timestamp", "processed_timestamp"]:
        ts = record.get(key)
        if ts and ts not in ["TIME_NOT_SYNCED", ""]:
            try:
                # Handle ISO formats
                return datetime.fromisoformat(ts[:10]).date()
            except Exception:
                pass
    return datetime.now().date()


def build_report_summary(records: list, period_type: str, period_label: str) -> dict:
    """
    Compute aggregate metrics, defect class counts, severity distribution,
    and road hazard hotspots from a list of detection records.
    """
    total_images = len(records)
    total_defects = 0
    defect_counts = {
        "pothole": 0,
        "longitudinal_crack": 0,
        "transverse_crack": 0,
        "alligator_crack": 0
    }
    severity_counts = {
        "Good": 0,
        "Fair": 0,
        "Poor": 0,
        "Critical": 0
    }
    camera_counts = defaultdict(lambda: {"images": 0, "defects": 0})
    total_score = 0.0
    gps_tagged_count = 0
    hotspots = []

    for r in records:
        defects = r.get("defect_count", 0)
        total_defects += defects
        sev = r.get("severity", "Good")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        score = r.get("severity_score", 0.0)
        total_score += score

        cam_id = r.get("camera_id", "CAM_001")
        camera_counts[cam_id]["images"] += 1
        camera_counts[cam_id]["defects"] += defects

        # Count defect types
        summary = r.get("summary", {})
        for k, v in summary.items():
            if k in defect_counts:
                defect_counts[k] += v

        # Hotspot / GPS tracking
        if r.get("gps_valid") and r.get("latitude") != "NA" and r.get("longitude") != "NA":
            gps_tagged_count += 1
            if sev in ["Critical", "Poor"] or defects > 0:
                hotspots.append({
                    "image": r.get("image_filename"),
                    "camera_id": cam_id,
                    "latitude": r.get("latitude"),
                    "longitude": r.get("longitude"),
                    "severity": sev,
                    "severity_score": score,
                    "defect_count": defects,
                    "potholes": summary.get("pothole", 0),
                    "cracks": summary.get("longitudinal_crack", 0) + summary.get("transverse_crack", 0) + summary.get("alligator_crack", 0),
                    "timestamp": r.get("capture_timestamp") or r.get("processed_timestamp")
                })

    avg_score = round(total_score / total_images, 1) if total_images > 0 else 0.0
    defect_density = round(total_defects / total_images, 2) if total_images > 0 else 0.0

    # Sort hotspots by severity score descending
    hotspots.sort(key=lambda h: h["severity_score"], reverse=True)

    # Simplified record list for report inclusion
    record_items = []
    for r in records:
        record_items.append({
            "image_filename": r.get("image_filename"),
            "camera_id": r.get("camera_id"),
            "timestamp": r.get("capture_timestamp") or r.get("processed_timestamp"),
            "severity": r.get("severity"),
            "severity_score": r.get("severity_score"),
            "defect_count": r.get("defect_count"),
            "summary": r.get("summary"),
            "gps_valid": r.get("gps_valid"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "processed_image_path": r.get("processed_image_path")
        })

    report = {
        "report_id": f"REP_{period_type.upper()}_{period_label.replace('-', '').replace('W', '_')}",
        "period_type": period_type,
        "period_label": period_label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total_images_analyzed": total_images,
        "total_defects_detected": total_defects,
        "defect_density_per_image": defect_density,
        "average_severity_score": avg_score,
        "overall_condition": (
            "Critical" if avg_score >= 60 or severity_counts["Critical"] > 2
            else "Poor" if avg_score >= 35 or severity_counts["Poor"] > 3
            else "Fair" if avg_score >= 15
            else "Good"
        ),
        "defects_by_class": defect_counts,
        "severity_breakdown": {
            k: {
                "count": v,
                "percentage": round((v / total_images) * 100, 1) if total_images > 0 else 0.0
            }
            for k, v in severity_counts.items()
        },
        "camera_breakdown": dict(camera_counts),
        "gps_tagged_images": gps_tagged_count,
        "critical_hotspots_count": len(hotspots),
        "hotspots": hotspots[:20],  # Top 20 hazard locations
        "records": record_items
    }

    return report


def export_report_csv(report: dict, output_csv_path: Path):
    """
    Export report records to a clean CSV format suitable for engineering & civil authorities.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "Image Filename", "Camera ID", "Timestamp", "Road Severity",
        "Severity Score", "Total Defects", "Potholes", "Longitudinal Cracks",
        "Transverse Cracks", "Alligator Cracks", "GPS Valid", "Latitude", "Longitude"
    ]

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for r in report.get("records", []):
            summary = r.get("summary") or {}
            writer.writerow([
                r.get("image_filename"),
                r.get("camera_id"),
                r.get("timestamp"),
                r.get("severity"),
                r.get("severity_score"),
                r.get("defect_count"),
                summary.get("pothole", 0),
                summary.get("longitudinal_crack", 0),
                summary.get("transverse_crack", 0),
                summary.get("alligator_crack", 0),
                r.get("gps_valid"),
                r.get("latitude"),
                r.get("longitude")
            ])


def generate_daily_report(target_date: date = None) -> dict:
    """Generate daily report for specified date (defaults to today)."""
    if target_date is None:
        target_date = datetime.now().date()

    all_records = get_all_detection_records()
    daily_records = [
        r for r in all_records
        if parse_record_date(r) == target_date
    ]

    date_str = target_date.isoformat()
    report = build_report_summary(daily_records, "daily", date_str)

    # Save JSON and CSV
    daily_dir = REPORTS_DIR / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    json_path = daily_dir / f"report_{date_str}.json"
    csv_path = daily_dir / f"report_{date_str}.csv"

    json_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    export_report_csv(report, csv_path)

    report["json_path"] = str(json_path.relative_to(BASE_DIR))
    report["csv_path"] = str(csv_path.relative_to(BASE_DIR))
    return report


def generate_weekly_report(target_date: date = None) -> dict:
    """Generate weekly report for the ISO week containing target_date."""
    if target_date is None:
        target_date = datetime.now().date()

    year, week_num, _ = target_date.isocalendar()
    week_label = f"{year}-W{week_num:02d}"

    all_records = get_all_detection_records()
    weekly_records = [
        r for r in all_records
        if parse_record_date(r).isocalendar()[:2] == (year, week_num)
    ]

    report = build_report_summary(weekly_records, "weekly", week_label)

    weekly_dir = REPORTS_DIR / "weekly"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    json_path = weekly_dir / f"report_{week_label}.json"
    csv_path = weekly_dir / f"report_{week_label}.csv"

    json_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    export_report_csv(report, csv_path)

    report["json_path"] = str(json_path.relative_to(BASE_DIR))
    report["csv_path"] = str(csv_path.relative_to(BASE_DIR))
    return report


def generate_monthly_report(target_date: date = None) -> dict:
    """Generate monthly report for the month containing target_date."""
    if target_date is None:
        target_date = datetime.now().date()

    month_label = target_date.strftime("%Y-%m")

    all_records = get_all_detection_records()
    monthly_records = [
        r for r in all_records
        if parse_record_date(r).strftime("%Y-%m") == month_label
    ]

    report = build_report_summary(monthly_records, "monthly", month_label)

    monthly_dir = REPORTS_DIR / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    json_path = monthly_dir / f"report_{month_label}.json"
    csv_path = monthly_dir / f"report_{month_label}.csv"

    json_path.write_text(json.dumps(report, indent=4), encoding="utf-8")
    export_report_csv(report, csv_path)

    report["json_path"] = str(json_path.relative_to(BASE_DIR))
    report["csv_path"] = str(csv_path.relative_to(BASE_DIR))
    return report


def list_saved_reports() -> list:
    """List all existing generated reports with metadata."""
    reports = []
    for rtype in ["daily", "weekly", "monthly"]:
        tdir = REPORTS_DIR / rtype
        if not tdir.exists():
            continue
        for jf in sorted(tdir.glob("report_*.json"), reverse=True):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                reports.append({
                    "report_id": data.get("report_id"),
                    "type": rtype,
                    "period_label": data.get("period_label"),
                    "generated_at": data.get("generated_at"),
                    "total_images": data.get("total_images_analyzed", 0),
                    "total_defects": data.get("total_defects_detected", 0),
                    "overall_condition": data.get("overall_condition", "Good"),
                    "json_file": jf.name,
                    "csv_file": jf.with_suffix(".csv").name,
                    "json_path": str(jf.relative_to(BASE_DIR)),
                    "csv_path": str(jf.with_suffix(".csv").relative_to(BASE_DIR))
                })
            except Exception as e:
                print(f"[Report] Error reading report {jf}: {e}")
    return reports


if __name__ == "__main__":
    print("[Reports] Generating current Daily, Weekly, and Monthly reports...")
    d_rep = generate_daily_report()
    w_rep = generate_weekly_report()
    m_rep = generate_monthly_report()
    print(f"Daily Report:   {d_rep['json_path']} ({d_rep['total_images_analyzed']} images, {d_rep['total_defects_detected']} defects)")
    print(f"Weekly Report:  {w_rep['json_path']} ({w_rep['total_images_analyzed']} images, {w_rep['total_defects_detected']} defects)")
    print(f"Monthly Report: {m_rep['json_path']} ({m_rep['total_images_analyzed']} images, {m_rep['total_defects_detected']} defects)")
