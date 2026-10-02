from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset"
)


# ============================================================
# SETTINGS
# ============================================================

SPLITS = [
    "train",
    "validation",
    "test"
]


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("FINAL DATASET SPLIT VERIFICATION")
print("=" * 60)


grand_total = 0


for split in SPLITS:

    split_path = DATASET_PATH / split

    print(f"\n{split.upper()}")
    print("-" * 60)

    split_total = 0

    for class_folder in sorted(split_path.iterdir()):

        if not class_folder.is_dir():
            continue

        image_count = len([
            image
            for image in class_folder.iterdir()
            if image.suffix.lower()
            in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        print(
            f"{class_folder.name:<25} : "
            f"{image_count} images"
        )

        split_total += image_count

    print(
        f"{'TOTAL':<25} : "
        f"{split_total} images"
    )

    grand_total += split_total


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(f"Total images across all splits : {grand_total}")

if grand_total == 1091:
    print("\n✅ DATASET VERIFICATION PASSED!")
else:
    print("\n⚠️ DATASET TOTAL DOES NOT MATCH 1091!")

print("=" * 60)