from pathlib import Path
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TEST_DIR = PROJECT_ROOT / "dataset" / "test"

YOLO11_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "runs"
    / "comparison"
    / "yolo11n"
    / "weights"
    / "best.pt"
)

YOLO26_MODEL = (
    PROJECT_ROOT
    / "runs"
    / "classify"
    / "runs"
    / "comparison"
    / "yolo26n"
    / "weights"
    / "best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "runs" / "comparison" / "ground_truth"

CSV_PATH = OUTPUT_DIR / "ground_truth_comparison.csv"
JSON_PATH = OUTPUT_DIR / "ground_truth_summary.json"

YOLO11_CM_PATH = OUTPUT_DIR / "yolo11n_confusion_matrix.png"
YOLO26_CM_PATH = OUTPUT_DIR / "yolo26n_confusion_matrix.png"


# Supported image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_test_images(test_dir):
    """
    Read test images and use their parent folder as the
    ground-truth class.
    """

    images = []

    for class_dir in sorted(test_dir.iterdir()):

        if not class_dir.is_dir():
            continue

        ground_truth = class_dir.name

        for image_path in sorted(class_dir.iterdir()):

            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(
                    {
                        "image_path": image_path,
                        "ground_truth": ground_truth,
                    }
                )

    return images


def predict_single_image(model, image_path):
    """
    Run classification on one image and return:
    predicted class and confidence.
    """

    results = model.predict(
        source=str(image_path),
        imgsz=224,
        verbose=False,
    )

    result = results[0]

    top1_index = result.probs.top1
    top1_confidence = float(result.probs.top1conf)

    predicted_class = result.names[top1_index]

    return predicted_class, top1_confidence


def calculate_per_class_accuracy(ground_truth, predictions, classes):
    """
    Calculate accuracy separately for every class.
    """

    results = {}

    for class_name in classes:

        total = 0
        correct = 0

        for actual, predicted in zip(ground_truth, predictions):

            if actual == class_name:

                total += 1

                if predicted == class_name:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        results[class_name] = {
            "total": total,
            "correct": correct,
            "incorrect": total - correct,
            "accuracy": accuracy,
        }

    return results


def save_confusion_matrix(
    ground_truth,
    predictions,
    classes,
    output_path,
    model_name,
):
    """
    Generate and save a confusion matrix.
    """

    matrix = confusion_matrix(
        ground_truth,
        predictions,
        labels=classes,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=classes,
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
    )

    ax.set_title(
        f"{model_name} - Ground Truth vs Prediction"
    )

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Ground Truth Class")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# ============================================================
# MAIN COMPARISON
# ============================================================

def main():

    print("=" * 80)
    print("GROUND-TRUTH MODEL COMPARISON")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{TEST_DIR}"
        )

    if not YOLO11_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO11n model not found:\n{YOLO11_MODEL}"
        )

    if not YOLO26_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO26n model not found:\n{YOLO26_MODEL}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Read test dataset
    # --------------------------------------------------------

    test_images = get_test_images(TEST_DIR)

    if not test_images:
        raise RuntimeError(
            "No test images were found."
        )

    classes = sorted(
        {
            item["ground_truth"]
            for item in test_images
        }
    )

    print()
    print(f"Test directory : {TEST_DIR}")
    print(f"Test images    : {len(test_images)}")
    print(f"Classes        : {classes}")

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    print()
    print("Loading YOLO11n-cls...")
    yolo11 = YOLO(str(YOLO11_MODEL))

    print("Loading YOLO26n-cls...")
    yolo26 = YOLO(str(YOLO26_MODEL))

    # --------------------------------------------------------
    # Prediction containers
    # --------------------------------------------------------

    ground_truths = []

    yolo11_predictions = []
    yolo11_confidences = []

    yolo26_predictions = []
    yolo26_confidences = []

    comparison_rows = []

    # --------------------------------------------------------
    # Predict every test image
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("RUNNING IMAGE-LEVEL COMPARISON")
    print("=" * 80)

    for index, item in enumerate(test_images, start=1):

        image_path = item["image_path"]
        ground_truth = item["ground_truth"]

        print(
            f"[{index}/{len(test_images)}] "
            f"{image_path.name} | "
            f"Ground Truth: {ground_truth}"
        )

        # YOLO11n
        yolo11_prediction, yolo11_confidence = predict_single_image(
            yolo11,
            image_path,
        )

        # YOLO26n
        yolo26_prediction, yolo26_confidence = predict_single_image(
            yolo26,
            image_path,
        )

        # Correct / incorrect
        yolo11_correct = (
            yolo11_prediction == ground_truth
        )

        yolo26_correct = (
            yolo26_prediction == ground_truth
        )

        ground_truths.append(ground_truth)

        yolo11_predictions.append(
            yolo11_prediction
        )

        yolo11_confidences.append(
            yolo11_confidence
        )

        yolo26_predictions.append(
            yolo26_prediction
        )

        yolo26_confidences.append(
            yolo26_confidence
        )

        comparison_rows.append(
            {
                "image": image_path.name,
                "ground_truth": ground_truth,
                "yolo11n_prediction": yolo11_prediction,
                "yolo11n_confidence": round(
                    yolo11_confidence,
                    4,
                ),
                "yolo11n_correct": yolo11_correct,
                "yolo26n_prediction": yolo26_prediction,
                "yolo26n_confidence": round(
                    yolo26_confidence,
                    4,
                ),
                "yolo26n_correct": yolo26_correct,
            }
        )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    yolo11_accuracy = accuracy_score(
        ground_truths,
        yolo11_predictions,
    )

    yolo26_accuracy = accuracy_score(
        ground_truths,
        yolo26_predictions,
    )

    yolo11_correct_count = sum(
        yolo11_predictions[i] == ground_truths[i]
        for i in range(len(ground_truths))
    )

    yolo26_correct_count = sum(
        yolo26_predictions[i] == ground_truths[i]
        for i in range(len(ground_truths))
    )

    yolo11_incorrect_count = (
        len(ground_truths) - yolo11_correct_count
    )

    yolo26_incorrect_count = (
        len(ground_truths) - yolo26_correct_count
    )

    # ========================================================
    # PER-CLASS ACCURACY
    # ========================================================

    yolo11_per_class = calculate_per_class_accuracy(
        ground_truths,
        yolo11_predictions,
        classes,
    )

    yolo26_per_class = calculate_per_class_accuracy(
        ground_truths,
        yolo26_predictions,
        classes,
    )

    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    save_confusion_matrix(
        ground_truths,
        yolo11_predictions,
        classes,
        YOLO11_CM_PATH,
        "YOLO11n-cls",
    )

    save_confusion_matrix(
        ground_truths,
        yolo26_predictions,
        classes,
        YOLO26_CM_PATH,
        "YOLO26n-cls",
    )

    # ========================================================
    # CLASSIFICATION REPORTS
    # ========================================================

    yolo11_report = classification_report(
        ground_truths,
        yolo11_predictions,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    yolo26_report = classification_report(
        ground_truths,
        yolo26_predictions,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        fieldnames = [
            "image",
            "ground_truth",
            "yolo11n_prediction",
            "yolo11n_confidence",
            "yolo11n_correct",
            "yolo26n_prediction",
            "yolo26n_confidence",
            "yolo26n_correct",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            comparison_rows
        )

    # ========================================================
    # SAVE JSON SUMMARY
    # ========================================================

    summary = {
        "dataset": {
            "test_directory": str(TEST_DIR),
            "test_images": len(test_images),
            "classes": classes,
        },
        "models": {
            "YOLO11n-cls": {
                "model_path": str(YOLO11_MODEL),
                "accuracy": yolo11_accuracy,
                "correct": yolo11_correct_count,
                "incorrect": yolo11_incorrect_count,
                "per_class": yolo11_per_class,
                "classification_report": yolo11_report,
            },
            "YOLO26n-cls": {
                "model_path": str(YOLO26_MODEL),
                "accuracy": yolo26_accuracy,
                "correct": yolo26_correct_count,
                "incorrect": yolo26_incorrect_count,
                "per_class": yolo26_per_class,
                "classification_report": yolo26_report,
            },
        },
        "outputs": {
            "csv": str(CSV_PATH),
            "yolo11n_confusion_matrix": str(YOLO11_CM_PATH),
            "yolo26n_confusion_matrix": str(YOLO26_CM_PATH),
        },
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ========================================================
    # PRINT FINAL RESULTS
    # ========================================================

    print()
    print("=" * 80)
    print("GROUND-TRUTH COMPARISON RESULTS")
    print("=" * 80)

    print()
    print(
        f"{'Metric':<30}"
        f"{'YOLO11n':>15}"
        f"{'YOLO26n':>15}"
    )

    print("-" * 60)

    print(
        f"{'Test Images':<30}"
        f"{len(test_images):>15}"
        f"{len(test_images):>15}"
    )

    print(
        f"{'Correct Predictions':<30}"
        f"{yolo11_correct_count:>15}"
        f"{yolo26_correct_count:>15}"
    )

    print(
        f"{'Incorrect Predictions':<30}"
        f"{yolo11_incorrect_count:>15}"
        f"{yolo26_incorrect_count:>15}"
    )

    print(
        f"{'Top-1 Accuracy':<30}"
        f"{yolo11_accuracy * 100:>14.2f}%"
        f"{yolo26_accuracy * 100:>14.2f}%"
    )

    print()
    print("=" * 80)
    print("PER-CLASS ACCURACY")
    print("=" * 80)

    print()
    print(
        f"{'Class':<25}"
        f"{'YOLO11n':>15}"
        f"{'YOLO26n':>15}"
    )

    print("-" * 55)

    for class_name in classes:

        acc11 = (
            yolo11_per_class[class_name]["accuracy"]
            * 100
        )

        acc26 = (
            yolo26_per_class[class_name]["accuracy"]
            * 100
        )

        print(
            f"{class_name:<25}"
            f"{acc11:>14.2f}%"
            f"{acc26:>14.2f}%"
        )

    print()
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(f"CSV:")
    print(CSV_PATH)

    print()
    print("YOLO11n Confusion Matrix:")
    print(YOLO11_CM_PATH)

    print()
    print("YOLO26n Confusion Matrix:")
    print(YOLO26_CM_PATH)

    print()
    print("Summary JSON:")
    print(JSON_PATH)

    print()
    print("=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()