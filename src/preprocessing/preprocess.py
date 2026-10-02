from pathlib import Path
from PIL import Image

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "dataset" / "Mulberry_Leaf"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "processed_dataset"

IMAGE_SIZE = (224, 224)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("IMAGE PREPROCESSING")
print("=" * 60)

for class_folder in DATASET_PATH.iterdir():

    if not class_folder.is_dir():
        continue

    output_class = OUTPUT_PATH / class_folder.name
    output_class.mkdir(exist_ok=True)

    images = list(class_folder.glob("*"))

    print(f"\nProcessing {class_folder.name}")

    for img_path in images:

        try:

            img = Image.open(img_path)

            img = img.convert("RGB")

            img = img.resize(IMAGE_SIZE)

            save_path = output_class / img_path.name

            img.save(save_path)

        except Exception as e:

            print("Skipped:", img_path.name)

print("\nDone!")