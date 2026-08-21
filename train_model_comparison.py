from ultralytics import YOLO
from pathlib import Path
import time
import json
import shutil

# ============================================================
# YOLO11n vs YOLO26n - Pizza Classification Comparison
# ============================================================

DATASET = Path("dataset")

IMAGE_SIZE = 224
EPOCHS = 20
BATCH_SIZE = 16
DEVICE = "cpu"

# Separate output directories for each experiment
YOLO11_RUN = "runs/comparison/yolo11n"
YOLO26_RUN = "runs/comparison/yolo26n"


def train_model(model_name, model_path, project_dir, run_name):
    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = YOLO(model_path)

    start_time = time.perf_counter()

    results = model.train(
        data=str(DATASET),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=project_dir,
        name=run_name,
        exist_ok=True,
        pretrained=True,
        verbose=True
    )

    training_time = time.perf_counter() - start_time

    return model, results, training_time


def validate_model(model, model_name, split="val"):
    print("\n" + "=" * 70)
    print(f"VALIDATING: {model_name}")
    print("=" * 70)

    metrics = model.val(
        data=str(DATASET),
        split=split,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        verbose=True
    )

    return metrics


def get_model_info(model):
    """
    Extract basic model information.
    """
    try:
        parameters = sum(
            p.numel()
            for p in model.model.parameters()
        )
    except Exception:
        parameters = None

    try:
        file_size_mb = Path(model.ckpt_path).stat().st_size / (1024 * 1024)
    except Exception:
        file_size_mb = None

    return {
        "parameters": parameters,
        "model_size_mb": file_size_mb
    }


def extract_accuracy(metrics):
    """
    Extract Top-1 and Top-5 accuracy from Ultralytics
    classification validation metrics.
    """
    top1 = None
    top5 = None

    try:
        top1 = float(metrics.top1)
    except Exception:
        pass

    try:
        top5 = float(metrics.top5)
    except Exception:
        pass

    return top1, top5


def main():

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET.resolve()}"
        )

    for split in ["train", "val", "test"]:
        split_path = DATASET / split

        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing dataset split: {split_path}"
            )

    print("\n" + "#" * 70)
    print("# YOLO11n vs YOLO26n PIZZA CLASSIFICATION")
    print("#" * 70)

    print(f"\nDataset       : {DATASET.resolve()}")
    print(f"Image size    : {IMAGE_SIZE}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Device        : {DEVICE}")

    # ========================================================
    # YOLO11n
    # ========================================================

    model11, results11, time11 = train_model(
        "YOLO11n-cls",
        "yolo11n-cls.pt",
        "runs/comparison",
        "yolo11n"
    )

    metrics11 = validate_model(
        model11,
        "YOLO11n-cls",
        split="test"
    )

    top1_11, top5_11 = extract_accuracy(metrics11)
    info11 = get_model_info(model11)

    # ========================================================
    # YOLO26n
    # ========================================================

    model26, results26, time26 = train_model(
        "YOLO26n-cls",
        "yolo26n-cls.pt",
        "runs/comparison",
        "yolo26n"
    )

    metrics26 = validate_model(
        model26,
        "YOLO26n-cls",
        split="test"
    )

    top1_26, top5_26 = extract_accuracy(metrics26)
    info26 = get_model_info(model26)

    # ========================================================
    # Comparison
    # ========================================================

    comparison = {
        "YOLO11n-cls": {
            "top1_accuracy": top1_11,
            "top5_accuracy": top5_11,
            "training_time_seconds": time11,
            "parameters": info11["parameters"],
            "model_size_mb": info11["model_size_mb"]
        },
        "YOLO26n-cls": {
            "top1_accuracy": top1_26,
            "top5_accuracy": top5_26,
            "training_time_seconds": time26,
            "parameters": info26["parameters"],
            "model_size_mb": info26["model_size_mb"]
        }
    }

    output_dir = Path("runs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=4)

    # ========================================================
    # Print final comparison
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Metric':<30}"
        f"{'YOLO11n-cls':<20}"
        f"{'YOLO26n-cls':<20}"
    )

    print("-" * 80)

    print(
        f"{'Top-1 Accuracy':<30}"
        f"{str(top1_11):<20}"
        f"{str(top1_26):<20}"
    )

    print(
        f"{'Top-5 Accuracy':<30}"
        f"{str(top5_11):<20}"
        f"{str(top5_26):<20}"
    )

    print(
        f"{'Training Time (seconds)':<30}"
        f"{time11:<20.2f}"
        f"{time26:<20.2f}"
    )

    print(
        f"{'Parameters':<30}"
        f"{str(info11['parameters']):<20}"
        f"{str(info26['parameters']):<20}"
    )

    print(
        f"{'Model Size (MB)':<30}"
        f"{str(info11['model_size_mb']):<20}"
        f"{str(info26['model_size_mb']):<20}"
    )

    print("=" * 80)

    print(f"\nResults saved to:")
    print(output_file.resolve())


if __name__ == "__main__":
    main()