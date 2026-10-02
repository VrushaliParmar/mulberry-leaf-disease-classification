from pathlib import Path
import shutil
import random


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DATASET = PROJECT_ROOT / "dataset" / "Mulberry_Leaf"

OUTPUT_DATASET = PROJECT_ROOT / "outputs" / "processed_dataset"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# CHECK RATIOS
# ============================================================

if abs(
    TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0
) > 0.001:

    raise ValueError(
        "Train, validation and test ratios must add up to 1."
    )


# ============================================================
# SET RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 60)
print("MULBERRY DATASET - STRATIFIED SPLIT")
print("=" * 60)


# ============================================================
# CHECK SOURCE DATASET
# ============================================================

if not SOURCE_DATASET.exists():

    raise FileNotFoundError(
        f"Dataset not found at:\n{SOURCE_DATASET}"
    )


print("\nSource dataset:")
print(SOURCE_DATASET)

print("\nOutput dataset:")
print(OUTPUT_DATASET)


# ============================================================
# REMOVE OLD PROCESSED DATASET
# ============================================================

if OUTPUT_DATASET.exists():

    print("\nRemoving previous processed dataset...")

    shutil.rmtree(OUTPUT_DATASET)


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

for split in ["train", "validation", "test"]:

    for class_folder in SOURCE_DATASET.iterdir():

        if class_folder.is_dir():

            output_class_folder = (
                OUTPUT_DATASET
                / split
                / class_folder.name
            )

            output_class_folder.mkdir(
                parents=True,
                exist_ok=True
            )


# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
}


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_images = 0

print("\n")
print("=" * 60)
print("SPLITTING DATASET")
print("=" * 60)


for class_folder in sorted(SOURCE_DATASET.iterdir()):

    if not class_folder.is_dir():
        continue


    # --------------------------------------------------------
    # GET IMAGES
    # --------------------------------------------------------

    images = [
        image
        for image in class_folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]


    # --------------------------------------------------------
    # SHUFFLE IMAGES
    # --------------------------------------------------------

    random.shuffle(images)


    # --------------------------------------------------------
    # CALCULATE SPLIT SIZES
    # --------------------------------------------------------

    total = len(images)

    train_count = int(total * TRAIN_RATIO)

    validation_count = int(
        total * VALIDATION_RATIO
    )

    test_count = (
        total
        - train_count
        - validation_count
    )


    # --------------------------------------------------------
    # SPLIT IMAGES
    # --------------------------------------------------------

    train_images = images[
        :train_count
    ]

    validation_images = images[
        train_count:
        train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]


    # --------------------------------------------------------
    # COPY IMAGES
    # --------------------------------------------------------

    splits = {
        "train": train_images,
        "validation": validation_images,
        "test": test_images
    }


    for split_name, split_images in splits.items():

        destination = (
            OUTPUT_DATASET
            / split_name
            / class_folder.name
        )


        for image in split_images:

            shutil.copy2(
                image,
                destination / image.name
            )


    # --------------------------------------------------------
    # PRINT CLASS RESULTS
    # --------------------------------------------------------

    print(
        f"\n{class_folder.name}"
    )

    print(
        f"Total      : {total}"
    )

    print(
        f"Training   : {len(train_images)}"
    )

    print(
        f"Validation : {len(validation_images)}"
    )

    print(
        f"Testing    : {len(test_images)}"
    )


    total_images += total


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)

print(
    f"\nTotal original images : {total_images}"
)

print(
    f"Train ratio           : {TRAIN_RATIO * 100:.0f}%"
)

print(
    f"Validation ratio      : {VALIDATION_RATIO * 100:.0f}%"
)

print(
    f"Test ratio            : {TEST_RATIO * 100:.0f}%"
)

print("\nDataset created at:")

print(OUTPUT_DATASET)

print("\n")
print("=" * 60)
print("DONE!")
print("=" * 60)