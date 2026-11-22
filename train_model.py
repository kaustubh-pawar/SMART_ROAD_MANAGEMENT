"""
Smart Road Management System - YOLO Model Training Script
Trains a custom YOLO road defect detection model on RDD2022 or custom dataset.
Supports MPS (Apple Silicon), CUDA, and CPU with automatic checkpoint installation.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = BASE_DIR / "datasets" / "RDD2022" / "data" / "data.yaml"
OUTPUT_MODEL_DIR = BASE_DIR / "AI_MODELS"
TARGET_MODEL_PATH = OUTPUT_MODEL_DIR / "pothole_crack.pt"


def detect_best_device():
    """Detect whether MPS (Apple Silicon), CUDA (NVIDIA), or CPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return "0"  # CUDA device 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def train_yolo(
    data_yaml: Path,
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 8,
    img_size: int = 640,
    device: str = None,
    project: str = "runs/road_defect",
    name: str = "rdd2022_custom"
):
    """Run YOLO training and copy best weights to AI_MODELS/pothole_crack.pt."""
    # Ensure temporary settings directory is writable
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")

    from ultralytics import YOLO

    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        print(f"[Error] Dataset configuration not found at: {data_path}")
        print("Please ensure the RDD2022 dataset has been downloaded to datasets/RDD2022/")
        print("Example download command:")
        print("  python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='ilyes2027/RDD2022', repo_type='dataset', local_dir='datasets/RDD2022')\"")
        return False

    if device is None or device == "auto":
        device = detect_best_device()

    print("\n==========================================")
    print("ROAD DEFECT YOLO TRAINING")
    print("==========================================")
    print(f"Dataset config: {data_path}")
    print(f"Base model:     {model_name}")
    print(f"Target device:  {device}")
    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Image size:     {img_size}")
    print("==========================================\n")

    print(f"[Training] Initializing base model: {model_name}...")
    model = YOLO(model_name)

    print("[Training] Starting training run...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=2,
        project=project,
        name=name,
        save=True
    )

    # Locate best.pt
    run_dir = Path(project) / name
    best_pt = run_dir / "weights" / "best.pt"

    if best_pt.exists():
        OUTPUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_pt, TARGET_MODEL_PATH)
        print("\n==========================================")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("==========================================")
        print(f"Best weights:   {best_pt}")
        print(f"Installed to:   {TARGET_MODEL_PATH} ({TARGET_MODEL_PATH.stat().st_size / (1024*1024):.2f} MB)")
        print("The detection engine will now use your newly trained custom weights!")
        print("==========================================\n")
        return True
    else:
        print(f"[Warning] Training finished, but best.pt was not found at {best_pt}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train custom YOLO road defect detector.")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base pretrained YOLO model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'mps', 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--project", type=str, default="runs/road_defect", help="Project output folder")
    parser.add_argument("--name", type=str, default="rdd2022_custom", help="Experiment run name")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name
    )


if __name__ == "__main__":
    main()
