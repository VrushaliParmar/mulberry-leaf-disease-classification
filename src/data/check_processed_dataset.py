from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "outputs" / "processed_dataset"


# ============================================================
# CHECK DATASET
# ============================================================

print("=" * 60)
print("PROCESSED DATASET CHECK")
print("=" * 60)

total = 0

for class_folder in sorted(DATASET_PATH.iterdir()):

    if not class_folder.is_dir():
        continue

    images = [
        file for file in class_folder.iterdir()
        if file.suffix.lower() in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        ]
    ]

    print(f"{class_folder.name} : {len(images)} images")

    total += len(images)


print("-" * 60)
print(f"TOTAL : {total} images")
print("=" * 60)