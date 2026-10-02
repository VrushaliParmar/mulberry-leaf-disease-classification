import tensorflow as tf
from pathlib import Path


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "outputs" / "processed_dataset"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 60)
print("TENSORFLOW DATASET SPLIT CHECK")
print("=" * 60)

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# COUNT LABELS
# ============================================================

def count_labels(dataset):

    counts = {}

    for _, labels in dataset:

        for label in labels.numpy():

            counts[int(label)] = counts.get(int(label), 0) + 1

    return counts


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLASS NAMES")
print("=" * 60)

for i, name in enumerate(train_dataset.class_names):
    print(f"{i} --> {name}")


train_counts = count_labels(train_dataset)

validation_counts = count_labels(validation_dataset)


print("\n" + "=" * 60)
print("TRAINING SET")
print("=" * 60)

for class_index, count in sorted(train_counts.items()):
    print(
        f"{train_dataset.class_names[class_index]} : {count}"
    )


print("\n" + "=" * 60)
print("VALIDATION SET")
print("=" * 60)

for class_index, count in sorted(validation_counts.items()):
    print(
        f"{validation_dataset.class_names[class_index]} : {count}"
    )


print("\n" + "=" * 60)
print("TOTALS")
print("=" * 60)

print(
    "Training images   :",
    sum(train_counts.values())
)

print(
    "Validation images :",
    sum(validation_counts.values())
)

print(
    "Total images      :",
    sum(train_counts.values())
    + sum(validation_counts.values())
)

print("=" * 60)