"""
Smart Road Management System - Image Processing & Defect Detection Engine
High-Accuracy Dual-Expert YOLO Ensemble + Multi-Scale Computer Vision Engine:
1. RDD2022 YOLOv8 Detector (Longitudinal Cracks, Transverse Cracks, Alligator Cracks, Potholes)
2. Pothole Specialist YOLOv8 Detector (High-Precision Cavity & Crater Detection, Water/Mud Filled)
3. Smart Defect Fusion & Non-Maximum Suppression (NMS)
4. Multi-Scale OpenCV Crack Fissure Glow & Surface Texture Extraction
5. Road Condition Severity Scoring (Good, Fair, Poor, Critical)
6. Professional HUD Image Annotation & Metadata Export
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
# Ensure headless environment flags before OpenCV / Torch load
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

try:
    import cv2
except ImportError as _cv_err:
    if "libGL.so.1" in str(_cv_err) or "shared object file" in str(_cv_err):
        import subprocess
        print(f"[OpenCV Loader] Caught shared library error ({_cv_err}). Auto-healing with opencv-python-headless...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-contrib-python"], check=False)
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "opencv-python-headless"], check=True)
            import cv2
            print("[OpenCV Loader] Successfully auto-healed cv2 import with opencv-python-headless!")
        except Exception as _heal_err:
            raise ImportError(f"OpenCV missing libGL.so.1 and auto-heal failed: {_heal_err}") from _cv_err
    else:
        raise

import numpy as np

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Ensure Ultralytics settings directory is writable
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

# AI Model Paths
RDD_MODEL_PATH = BASE_DIR / "AI_MODELS" / "rdd2022_yolov8s.pt"
POTHOLE_MODEL_PATH = BASE_DIR / "AI_MODELS" / "pothole_detection_yolov8.pt"
FALLBACK_MODEL_PATH = BASE_DIR / "AI_MODELS" / "pothole_crack.pt"

# Defect class mapping per RDD2022 standards
CLASSES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "alligator_crack",
    3: "pothole"
}

# Color palette for classes (BGR for OpenCV)
CLASS_COLORS = {
    "pothole": (34, 34, 220),           # Crimson Red
    "alligator_crack": (0, 140, 255),    # Vibrant Orange
    "longitudinal_crack": (0, 215, 255), # Golden Yellow
    "transverse_crack": (230, 216, 0),   # Cyan / Light Blue
    "unknown": (180, 180, 180)          # Gray
}

# Severity categories
SEVERITY_LEVELS = {
    "GOOD": {"label": "Good", "color": (46, 204, 113), "max_score": 15},
    "FAIR": {"label": "Fair", "color": (52, 152, 219), "max_score": 40},
    "POOR": {"label": "Poor", "color": (243, 156, 18), "max_score": 70},
    "CRITICAL": {"label": "Critical", "color": (231, 76, 60), "max_score": 100}
}


def is_valid_model_file(model_path: Path) -> bool:
    """Check if model file exists and is greater than 1MB."""
    try:
        return model_path.exists() and model_path.stat().st_size > 1024 * 1024
    except Exception:
        return False


class DefectDetector:
    """
    High-Accuracy Road Defect Detection System:
    - Primary Engine: Dual-Expert YOLO Ensemble (RDD2022 Benchmark + Specialized Pothole Model)
    - Fallback Engine: Advanced Multi-Scale OpenCV Computer Vision Analyzer
    - Surface Glow: Generates fine-grained crack fissure illumination mask
    """

    def __init__(self):
        self.rdd_model = None
        self.pothole_model = None
        self.mode = "cv_heuristic"
        self._load_models()

    def _load_models(self):
        try:
            from ultralytics import YOLO

            # 1. Load RDD2022 Model
            rdd_target = RDD_MODEL_PATH if is_valid_model_file(RDD_MODEL_PATH) else FALLBACK_MODEL_PATH
            if is_valid_model_file(rdd_target):
                try:
                    self.rdd_model = YOLO(str(rdd_target))
                    print(f"[Detector] Loaded RDD2022 model from {rdd_target.name}")
                except Exception as e:
                    print(f"[Detector] Failed loading RDD model: {e}")

            # 2. Load Pothole Specialist Model
            if is_valid_model_file(POTHOLE_MODEL_PATH):
                try:
                    self.pothole_model = YOLO(str(POTHOLE_MODEL_PATH))
                    print(f"[Detector] Loaded Pothole specialist model from {POTHOLE_MODEL_PATH.name}")
                except Exception as e:
                    print(f"[Detector] Failed loading Pothole model: {e}")

            # Determine operating mode
            if self.rdd_model is not None and self.pothole_model is not None:
                self.mode = "ensemble_yolo"
            elif self.rdd_model is not None:
                self.mode = "rdd_yolo"
            elif self.pothole_model is not None:
                self.mode = "pothole_yolo"
            else:
                self.mode = "cv_heuristic"
                print("[Detector] No valid YOLO models found. Operating in CV heuristic mode.")

        except ImportError:
            print("[Detector] Ultralytics YOLO not installed. Operating in CV heuristic mode.")
            self.mode = "cv_heuristic"

    def detect(self, image: np.ndarray) -> tuple:
        """
        Run defect detection on input BGR image.
        Returns:
            detections (list): list of defect dicts
            crack_mask (np.ndarray): binary mask of crack fissures for visual glow
        """
        if self.mode != "cv_heuristic":
            try:
                return self._detect_yolo_ensemble(image)
            except Exception as e:
                print(f"[Detector] YOLO inference error ({e}). Falling back to CV detector.")
                return self._detect_cv_heuristics(image)

        return self._detect_cv_heuristics(image)

    def _detect_yolo_ensemble(self, image: np.ndarray) -> tuple:
        h, w = image.shape[:2]
        candidates = []

        # -------------------------------------------------------------
        # 1. Run RDD2022 Road Damage Model (Cracks & Potholes)
        # -------------------------------------------------------------
        if self.rdd_model is not None:
            r_rdd = self.rdd_model.predict(image, conf=0.22, verbose=False)[0]
            for b in r_rdd.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                name = self.rdd_model.names.get(cls_id, CLASSES.get(cls_id, f"defect_{cls_id}"))
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                xyxy = [max(0, xyxy[0]), max(0, xyxy[1]), min(w - 1, xyxy[2]), min(h - 1, xyxy[3])]
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                if bw <= 0 or bh <= 0:
                    continue

                candidates.append({
                    "class": name,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": xyxy,
                    "area_pixels": int(bw * bh),
                    "source": "rdd"
                })

        # -------------------------------------------------------------
        # 2. Run Pothole Specialist Model (High-Precision Cavities)
        # -------------------------------------------------------------
        if self.pothole_model is not None:
            r_pot = self.pothole_model.predict(image, conf=0.35, verbose=False)[0]
            for b in r_pot.boxes:
                conf = float(b.conf[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                xyxy = [max(0, xyxy[0]), max(0, xyxy[1]), min(w - 1, xyxy[2]), min(h - 1, xyxy[3])]
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                if bw <= 0 or bh <= 0:
                    continue

                aspect = float(bw) / float(bh) if bh > 0 else 1.0
                width_ratio = float(bw) / float(w)

                # Geometry check: if the bounding box spans > 78% of the road width
                # with an extreme aspect ratio (> 2.5), it represents road-wide fatigue cracking
                if width_ratio > 0.78 and aspect > 2.5:
                    cls_name = "alligator_crack"
                else:
                    cls_name = "pothole"

                candidates.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "bbox_xyxy": xyxy,
                    "area_pixels": int(bw * bh),
                    "source": "pothole_expert"
                })

        # -------------------------------------------------------------
        # 3. Smart Defect Fusion & Suppression
        # -------------------------------------------------------------
        detections = self._fuse_defect_candidates(candidates)

        # -------------------------------------------------------------
        # 4. Fallback Verification (Safety net for subtle fissures)
        # -------------------------------------------------------------
        if not detections:
            cv_dets, cv_mask = self._detect_cv_heuristics(image)
            # Only accept high-confidence CV detections if YOLO was clean
            high_conf_cv = [d for d in cv_dets if d.get("confidence", 0) >= 0.85]
            if high_conf_cv:
                detections = high_conf_cv

        # -------------------------------------------------------------
        # 5. Extract Crack Glow Mask for Visualization
        # -------------------------------------------------------------
        crack_mask = self._extract_crack_illumination_mask(image, detections)

        return detections, crack_mask

    def _fuse_defect_candidates(self, candidates: list) -> list:
        if not candidates:
            return []

        # Ranking score:
        # A confirmed pothole (conf >= 0.50) receives a priority bonus
        # because a physical cavity / crater is the primary structural hazard
        def score(d):
            base = d["confidence"]
            if d["class"] == "pothole" and base >= 0.50:
                return base + 0.25
            return base

        dets = sorted(candidates, key=score, reverse=True)
        keep = []

        while dets:
            curr = dets.pop(0)
            keep.append(curr)
            rem = []
            box_a = curr["bbox_xyxy"]

            for other in dets:
                box_b = other["bbox_xyxy"]
                x_left = max(box_a[0], box_b[0])
                y_top = max(box_a[1], box_b[1])
                x_right = min(box_a[2], box_b[2])
                y_bottom = min(box_a[3], box_b[3])

                if x_right > x_left and y_bottom > y_top:
                    intersection = (x_right - x_left) * (y_bottom - y_top)
                    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
                    union = float(area_a + area_b - intersection)
                    iou = intersection / union if union > 0 else 0
                    overlap_b = intersection / float(area_b) if area_b > 0 else 0
                    overlap_a = intersection / float(area_a) if area_a > 0 else 0
                else:
                    iou = 0
                    overlap_b = 0
                    overlap_a = 0

                # If current detection is a strong pothole (conf >= 0.65) and the other is a faint
                # alligator crack box enclosing that same crater, suppress the redundant crack box
                if curr["class"] == "pothole" and curr["confidence"] >= 0.65 and other["class"] == "alligator_crack" and other["confidence"] < 0.50 and overlap_a >= 0.80:
                    continue

                # Suppress if heavy IoU overlap (same defect) or if candidate is >= 70% contained
                if iou >= 0.40 or overlap_b >= 0.70:
                    continue

                rem.append(other)

            dets = rem

        # Format final detections
        result = []
        for d in keep:
            result.append({
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox_xyxy": d["bbox_xyxy"],
                "area_pixels": d["area_pixels"]
            })
        return result

    def _extract_crack_illumination_mask(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Extract illuminated crack fissures across road surface for high-contrast HUD display."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Bilateral filter suppresses fine asphalt aggregate while preserving sharp fissures
        filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=35)

        # Black-Hat transform highlights thin dark channels
        bh = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        sobelx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        comb = cv2.bitwise_and(bh, grad)

        _, th_c = cv2.threshold(comb, 22, 255, cv2.THRESH_BINARY)
        crack_mask = cv2.medianBlur(th_c, 3)

        # Restrict glow to lower road region
        roi_top = int(h * 0.25)
        crack_mask[:roi_top, :] = 0

        # Also mask out watermark strips if present at the bottom
        if gray[-15:, :].mean() < 22:
            crack_mask[int(h * 0.94):, :] = 0

        return crack_mask

    def _detect_cv_heuristics(self, image: np.ndarray) -> tuple:
        """
        Multi-Scale Road Defect Detection Algorithm (Safety-Net Fallback):
        - Focuses on the road pavement surface (excludes sky/tree foliage)
        - Illumination normalization via CLAHE
        - Fissure extraction via combined Black-Hat transform and gradient magnitude
        - Clustered alligator crack detection spanning fragmented pavement blocks
        - Prominent linear crack detection (longitudinal & transverse)
        - Depressed cavity detection (potholes)
        """
        h, w = image.shape[:2]
        detections = []

        # 1. Road Region of Interest (ROI) & Watermark Suppression
        gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bottom_y = h
        if gray_full[-15:, :].mean() < 22:
            bottom_y = int(h * 0.94)

        roi_top = int(h * 0.28)
        road_gray = gray_full[roi_top:bottom_y, :]
        rh, rw = road_gray.shape

        # Filter out completely underexposed / dark lens-cap frames
        if road_gray.mean() < 22 and road_gray.std() < 16:
            return [], None

        total_road_pixels = rh * rw

        # 2. Bilateral filter
        filtered = cv2.bilateralFilter(road_gray, d=7, sigmaColor=35, sigmaSpace=35)

        # 3. Pothole Detection (Depressed localized cavities)
        bg = cv2.GaussianBlur(filtered, (45, 45), 0)
        depression = cv2.subtract(bg, filtered)
        _, dep_mask = cv2.threshold(depression, 22, 255, cv2.THRESH_BINARY)
        dep_mask = cv2.medianBlur(dep_mask, 5)

        crater_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        crater_closed = cv2.morphologyEx(dep_mask, cv2.MORPH_CLOSE, crater_kernel)
        pothole_contours, _ = cv2.findContours(crater_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pothole_mask = np.zeros((rh, rw), dtype=np.uint8)

        for cnt in pothole_contours:
            p_area = cv2.contourArea(cnt)
            if p_area < (total_road_pixels * 0.015):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh) if bh > 0 else 1.0
            if aspect > 2.6 or aspect < 0.38:
                continue

            patch = road_gray[by:by+bh, bx:bx+bw]
            darkness = road_gray.mean() - patch.mean()
            if darkness > 12:
                conf = min(0.96, 0.82 + (darkness / road_gray.mean()) * 0.4)
                detections.append({
                    "class": "pothole",
                    "confidence": round(float(conf), 2),
                    "bbox_xyxy": [bx, roi_top + by, bx + bw, roi_top + by + bh],
                    "area_pixels": int(bw * bh)
                })
                cv2.drawContours(pothole_mask, [cnt], -1, 255, -1)

        # 4. Crack Extraction
        bh = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
        sobelx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.uint8(np.clip(cv2.magnitude(sobelx, sobely), 0, 255))
        comb = cv2.bitwise_and(bh, grad)
        _, th_c = cv2.threshold(comb, 22, 255, cv2.THRESH_BINARY)
        crack_mask = cv2.medianBlur(th_c, 3)
        crack_mask_outside = cv2.bitwise_and(crack_mask, cv2.bitwise_not(pothole_mask))

        full_crack_mask = np.zeros((h, w), dtype=np.uint8)
        full_crack_mask[roi_top:bottom_y, :] = crack_mask

        # 5. Alligator Crack Surface Fatigue
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        closed_cracks = cv2.morphologyEx(crack_mask_outside, cv2.MORPH_CLOSE, close_k)
        fatigue_contours, _ = cv2.findContours(closed_cracks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in fatigue_contours:
            area = cv2.contourArea(cnt)
            if area < (total_road_pixels * 0.035):
                continue
            abx, aby, abw, abh = cv2.boundingRect(cnt)
            patch_cracks = crack_mask_outside[aby:aby+abh, abx:abx+abw]
            patch_density = np.count_nonzero(patch_cracks) / (abw * abh)
            if patch_density >= 0.09:
                conf = min(0.96, 0.75 + patch_density * 0.8)
                detections.append({
                    "class": "alligator_crack",
                    "confidence": round(float(conf), 2),
                    "bbox_xyxy": [abx, roi_top + aby, abx + abw, roi_top + aby + abh],
                    "area_pixels": int(abw * abh)
                })

        # 6. Prominent Linear Cracks
        raw_contours, _ = cv2.findContours(crack_mask_outside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_line_len = max(30, int(min(rh, rw) * 0.10))

        for cnt in raw_contours:
            if cv2.contourArea(cnt) < 90:
                continue
            bx, by, bw, bh_box = cv2.boundingRect(cnt)
            aspect = float(bw) / float(bh_box) if bh_box > 0 else 1.0

            center_x = bx + bw // 2
            center_y = roi_top + by + bh_box // 2
            inside_existing = False
            for d in detections:
                b = d["bbox_xyxy"]
                if b[0] <= center_x <= b[2] and b[1] <= center_y <= b[3]:
                    inside_existing = True
                    break
            if inside_existing:
                continue

            if (aspect < 0.40) and (bh_box >= min_line_len):
                conf = min(0.94, 0.74 + min(0.18, (1.0 / (aspect + 0.1)) * 0.03))
                detections.append({
                    "class": "longitudinal_crack",
                    "confidence": round(float(conf), 2),
                    "bbox_xyxy": [max(0, bx - 3), roi_top + max(0, by - 3), min(w - 1, bx + bw + 3), roi_top + min(rh - 1, by + bh_box + 3)],
                    "area_pixels": int(bw * bh_box)
                })
            elif (aspect > 2.5) and (bw >= min_line_len):
                conf = min(0.94, 0.74 + min(0.18, aspect * 0.03))
                detections.append({
                    "class": "transverse_crack",
                    "confidence": round(float(conf), 2),
                    "bbox_xyxy": [max(0, bx - 3), roi_top + max(0, by - 3), min(w - 1, bx + bw + 3), roi_top + min(rh - 1, by + bh_box + 3)],
                    "area_pixels": int(bw * bh_box)
                })

        return detections, full_crack_mask


# Global detector singleton
_DETECTOR = None

def get_detector() -> DefectDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DefectDetector()
    return _DETECTOR


def calculate_severity(detections: list, image_shape: tuple, crack_mask: np.ndarray = None) -> tuple:
    """
    Calculate road condition severity category and numerical score (0 - 100).
    Properly weights defect hazards:
    - Pothole: 42.0 points (severe vehicle impact & blowout hazard)
    - Alligator crack: 28.0 points (base pavement fatigue failure)
    - Longitudinal crack: 16.0 points (linear fissuring)
    - Transverse crack: 16.0 points (thermal/joint fissuring)
    """
    if not detections and (crack_mask is None or np.count_nonzero(crack_mask) == 0):
        return "Good", 0.0

    h, w = image_shape[:2]
    total_image_pixels = h * w

    CLASS_WEIGHTS = {
        "pothole": 42.0,
        "alligator_crack": 28.0,
        "longitudinal_crack": 16.0,
        "transverse_crack": 16.0
    }

    base_score = 0.0
    for d in detections:
        cls_name = d.get("class", "unknown")
        conf = d.get("confidence", 0.7)
        weight = CLASS_WEIGHTS.get(cls_name, 12.0)
        base_score += weight * conf

    # Surface crack coverage penalty
    coverage_score = 0.0
    if crack_mask is not None:
        crack_px = np.count_nonzero(crack_mask)
        coverage_ratio = crack_px / (total_image_pixels + 1e-5)
        coverage_score = min(35.0, coverage_ratio * 140.0)

    raw_score = base_score + coverage_score
    normalized_score = min(100.0, max(0.0, raw_score))

    # Category determination
    pothole_count = sum(1 for d in detections if d.get("class") == "pothole")
    alligator_count = sum(1 for d in detections if d.get("class") == "alligator_crack")

    if pothole_count >= 2 or alligator_count >= 3 or (pothole_count >= 1 and alligator_count >= 1) or normalized_score >= 60:
        severity = "Critical"
    elif pothole_count >= 1 or alligator_count >= 1 or normalized_score >= 35:
        severity = "Poor"
    elif len(detections) >= 1 or normalized_score >= 14:
        severity = "Fair"
    else:
        severity = "Good"

    return severity, round(normalized_score, 1)


def annotate_image(
    image: np.ndarray,
    detections: list,
    severity: str,
    severity_score: float,
    metadata: dict = None,
    crack_mask: np.ndarray = None
) -> np.ndarray:
    """
    Draw professional detection bounding boxes, illuminated crack fissures,
    and a sleek modern top HUD banner on the image.
    """
    annotated = image.copy()
    h, w = annotated.shape[:2]

    # 1. Overlay illuminated crack fissures directly onto the road surface
    if crack_mask is not None and np.count_nonzero(crack_mask) > 0:
        dilated_mask = cv2.dilate(crack_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        glow_color = np.zeros_like(annotated)
        glow_color[dilated_mask > 0] = [0, 140, 255] # High-contrast Amber / Orange glow

        alpha = 0.55
        annotated[dilated_mask > 0] = cv2.addWeighted(
            annotated[dilated_mask > 0], 1.0 - alpha,
            glow_color[dilated_mask > 0], alpha, 0
        )

    # 2. Draw each defect bounding box with corner styling and label pills
    for d in detections:
        cls_name = d.get("class", "defect")
        conf = d.get("confidence", 0.0)
        box = d.get("bbox_xyxy", [0, 0, 0, 0])
        x1, y1, x2, y2 = box

        color = CLASS_COLORS.get(cls_name, (0, 255, 0))

        # Main bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Sleek corner accents
        corner_len = min(18, max(8, int((x2 - x1) * 0.15)))
        t = 3
        # Top-left
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, t)
        # Top-right
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, t)
        # Bottom-left
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, t)
        # Bottom-right
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, t)

        # Label pill
        label = f"{cls_name.replace('_', ' ').title()}: {int(conf * 100)}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )
        pill_y1 = max(0, y1 - text_h - 8)
        pill_y2 = y1
        pill_x2 = min(w, x1 + text_w + 10)

        # Pill background
        cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
        # Pill text
        text_color = (0, 0, 0) if cls_name in ["longitudinal_crack", "transverse_crack"] else (255, 255, 255)
        cv2.putText(
            annotated, label, (x1 + 5, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA
        )

    # 3. Top HUD Banner
    banner_height = 42
    banner_overlay = annotated.copy()
    cv2.rectangle(banner_overlay, (0, 0), (w, banner_height), (18, 22, 30), -1)
    cv2.addWeighted(banner_overlay, 0.88, annotated, 0.12, 0, annotated)

    # Severity Badge Color
    sev_info = SEVERITY_LEVELS.get(severity.upper(), SEVERITY_LEVELS["GOOD"])
    sev_color = sev_info["color"]

    # Severity Badge on left
    badge_w = 95
    cv2.rectangle(annotated, (8, 7), (8 + badge_w, banner_height - 7), sev_color, -1)
    sev_text = severity.upper()
    (tw, th), _ = cv2.getTextSize(sev_text, cv2.FONT_HERSHEY_DUPLEX, 0.48, 1)
    tx = 8 + (badge_w - tw) // 2
    cv2.putText(
        annotated, sev_text, (tx, 26),
        cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
    )

    # Defect summary text
    defect_count = len(detections)
    info_text = f"Defects: {defect_count} | Score: {severity_score}/100"
    cv2.putText(
        annotated, info_text, (8 + badge_w + 12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 235, 245), 1, cv2.LINE_AA
    )

    # Right side: Camera ID & GPS
    cam_id = (metadata.get("camera_id") if metadata else None) or "CAM_001"
    lat = metadata.get("latitude", "NA") if metadata else "NA"
    lng = metadata.get("longitude", "NA") if metadata else "NA"
    gps_str = f"GPS: {lat[:7]}, {lng[:7]}" if lat != "NA" and lng != "NA" else "GPS: None"
    right_text = f"{cam_id} | {gps_str}"
    (rtw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)

    if w - rtw - 12 > (8 + badge_w + 160):
        cv2.putText(
            annotated, right_text, (w - rtw - 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 185, 205), 1, cv2.LINE_AA
        )

    return annotated


def process_single_image(image_path: Path, metadata_path: Path = None) -> dict:
    """
    Process a single road image:
    1. Reads original image with OpenCV
    2. Runs defect detection (Dual-Expert YOLO Ensemble)
    3. Calculates severity condition and score
    4. Annotates image with boxes + illuminated crack glow
    5. Writes annotated image to 'processed/' and detection JSON to 'detections/'
    6. Updates metadata JSON
    Returns the detection result dictionary.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read original image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")

    # Load metadata if exists
    metadata = {}
    if metadata_path is None:
        candidate_meta = image_path.parent.parent / "metadata" / (image_path.stem + ".json")
        if candidate_meta.exists():
            metadata_path = candidate_meta

    if metadata_path and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[Warning] Failed to read metadata at {metadata_path}: {e}")

    # Run detection
    detector = get_detector()
    detections, crack_mask = detector.detect(image)

    # Compute road severity
    severity, severity_score = calculate_severity(detections, image.shape, crack_mask)

    # Class summary counts
    summary = {
        "pothole": 0,
        "longitudinal_crack": 0,
        "transverse_crack": 0,
        "alligator_crack": 0
    }
    for d in detections:
        cls_name = d.get("class", "unknown")
        if cls_name in summary:
            summary[cls_name] += 1
        else:
            summary[cls_name] = summary.get(cls_name, 0) + 1

    # Prepare directories
    parent_dir = image_path.parent
    if parent_dir.name == "original":
        day_dir = parent_dir.parent
        processed_dir = day_dir / "processed"
        detections_dir = day_dir / "detections"
    else:
        processed_dir = parent_dir / "processed"
        detections_dir = parent_dir / "detections"

    processed_dir.mkdir(parents=True, exist_ok=True)
    detections_dir.mkdir(parents=True, exist_ok=True)

    # Generate annotated image with illuminated cracks
    annotated = annotate_image(image, detections, severity, severity_score, metadata, crack_mask)
    processed_image_path = processed_dir / image_path.name
    cv2.imwrite(str(processed_image_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Structure detection result
    now_iso = datetime.now().isoformat(timespec="seconds")
    result = {
        "camera_id": metadata.get("camera_id", "CAM_001"),
        "image_filename": image_path.name,
        "original_image_path": str(image_path.relative_to(BASE_DIR)) if image_path.is_relative_to(BASE_DIR) else str(image_path),
        "processed_image_path": str(processed_image_path.relative_to(BASE_DIR)) if processed_image_path.is_relative_to(BASE_DIR) else str(processed_image_path),
        "capture_timestamp": metadata.get("capture_timestamp", ""),
        "processed_timestamp": now_iso,
        "latitude": metadata.get("latitude", "NA"),
        "longitude": metadata.get("longitude", "NA"),
        "gps_valid": metadata.get("gps_valid", False),
        "detector_engine": detector.mode,
        "defect_count": len(detections),
        "severity": severity,
        "severity_score": severity_score,
        "summary": summary,
        "detections": detections
    }

    # Save detection JSON
    detection_json_path = detections_dir / (image_path.stem + ".json")
    detection_json_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
    result["detection_json_path"] = str(detection_json_path.relative_to(BASE_DIR)) if detection_json_path.is_relative_to(BASE_DIR) else str(detection_json_path)

    # Update metadata JSON
    if metadata_path and metadata_path.exists():
        try:
            metadata["processing_status"] = "completed"
            metadata["processed_timestamp"] = now_iso
            metadata["severity"] = severity
            metadata["severity_score"] = severity_score
            metadata["defect_count"] = len(detections)
            metadata["processed_image_path"] = result["processed_image_path"]
            metadata["detection_json_path"] = result["detection_json_path"]
            metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
        except Exception as e:
            print(f"[Warning] Failed to update metadata: {e}")

    return result


def batch_process_all_pending(base_dir: Path = BASE_DIR, force: bool = False) -> list:
    """
    Find and process all captured images in the workspace.
    If force=True, re-processes images even if already processed.
    """
    base_dir = Path(base_dir)
    original_images = list(base_dir.glob("**/original/*.jpg")) + list(base_dir.glob("**/original/*.jpeg")) + list(base_dir.glob("**/original/*.png"))
    results = []

    print(f"\n[Batch Processor] Found {len(original_images)} image(s).")
    for img_path in sorted(original_images):
        stem = img_path.stem
        det_file = img_path.parent.parent / "detections" / f"{stem}.json"
        proc_file = img_path.parent.parent / "processed" / img_path.name

        if not force and det_file.exists() and proc_file.exists():
            continue

        try:
            print(f"[Batch Processor] Processing: {img_path.name}...")
            res = process_single_image(img_path)
            results.append(res)
            print(f"  -> Detected {res['defect_count']} defect(s) | Severity: {res['severity']} ({res['severity_score']}/100)")
        except Exception as e:
            print(f"  -> Error processing {img_path.name}: {e}")

    print(f"[Batch Processor] Done. Processed {len(results)} image(s).\n")
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ["--all", "-a", "all"]:
            force_flag = "--force" in sys.argv or "-f" in sys.argv
            batch_process_all_pending(force=force_flag)
        else:
            p = Path(arg)
            if p.is_file():
                res = process_single_image(p)
                print(json.dumps(res, indent=2))
            elif p.is_dir():
                batch_process_all_pending(p, force=True)
    else:
        batch_process_all_pending()
