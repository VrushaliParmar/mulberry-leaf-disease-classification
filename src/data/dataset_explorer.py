from pathlib import Path
from PIL import Image
from collections import Counter

DATASET_PATH = Path("dataset/Mulberry_Leaf")

print("=" * 60)
print("      MULBERRY DATASET ANALYSIS")
print("=" * 60)

total_images = 0
formats = Counter()
sizes = Counter()
corrupted = []

for folder in DATASET_PATH.iterdir():

    if not folder.is_dir():
        continue

    print(f"\n📁 Class: {folder.name}")

    images = list(folder.glob("*"))

    print(f"Images: {len(images)}")

    total_images += len(images)

    for img_path in images:

        try:

            with Image.open(img_path) as img:

                formats[img.format] += 1

                sizes[img.size] += 1

        except Exception:

            corrupted.append(img_path.name)

print("\n" + "=" * 60)

print(f"Total Images : {total_images}")

print("\nImage Formats")

for f, c in formats.items():
    print(f"{f} : {c}")

print("\nMost Common Image Sizes")

for size, count in sizes.most_common(10):
    print(size, ":", count)

print("\nCorrupted Images :", len(corrupted))

if corrupted:
    print(corrupted)

print("=" * 60)