from pathlib import Path
import random
import shutil

# -----------------------------
# Configuration
# -----------------------------

random.seed(42)

SOURCE_CLASSES = {
    "bianca": Path("Bianca dataset/train/bianca"),
    "pepporoni": Path("Pepparoni dataset (1)/train/pepporoni"),
    "hawaiian": Path("hawaiin dataset/train/hawaiian"),
    "manager_choice": Path("manager_choice dataset/train/manager_choice"),
}

OUTPUT_ROOT = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# -----------------------------
# Validation
# -----------------------------

if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-9:
    raise ValueError("Train, validation and test ratios must add up to 1.0.")


# -----------------------------
# Create output directories
# -----------------------------

for split in ["train", "val", "test"]:
    for class_name in SOURCE_CLASSES:
        (OUTPUT_ROOT / split / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


# -----------------------------
# Split and copy images
# -----------------------------

total_copied = 0

for class_name, source_dir in SOURCE_CLASSES.items():

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Source folder not found: {source_dir}"
        )

    images = [
        file
        for file in source_dir.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not images:
        raise ValueError(
            f"No images found in: {source_dir}"
        )

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)
    test_count = total - train_count - val_count

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images,
    }

    print(f"\nClass: {class_name}")
    print(f"Total: {total}")
    print(f"Train: {len(train_images)}")
    print(f"Val:   {len(val_images)}")
    print(f"Test:  {len(test_images)}")

    for split_name, split_images in splits.items():

        destination_dir = OUTPUT_ROOT / split_name / class_name

        for image in split_images:
            destination = destination_dir / image.name

            shutil.copy2(image, destination)

            total_copied += 1


# -----------------------------
# Final summary
# -----------------------------

print("\n" + "=" * 50)
print("DATASET SPLIT COMPLETE")
print("=" * 50)

print(f"Total images copied: {total_copied}")

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in SOURCE_CLASSES:

        class_dir = OUTPUT_ROOT / split / class_name

        count = sum(
            1
            for file in class_dir.iterdir()
            if file.is_file()
            and file.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"{class_name}: {count}")

        split_total += count

    print(f"Total: {split_total}")