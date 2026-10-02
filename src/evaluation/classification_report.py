import tensorflow as tf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "saved_models" / "final_model.keras"

DATASET_PATH = PROJECT_ROOT / "outputs" / "processed_dataset"

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "plots"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("MULBERRY DISEASE CLASSIFICATION REPORT")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = validation_dataset.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index} --> {class_name}")


# ============================================================
# GET TRUE LABELS
# ============================================================

print("\nCollecting true labels...")

y_true = []

for images, labels in validation_dataset:
    y_true.extend(labels.numpy())

y_true = np.array(y_true)


# ============================================================
# GET PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    validation_dataset,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    digits=4
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    y_true,
    y_pred
)

print(cm)


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    xticks_rotation=45
)

plt.title("Mulberry Leaf Disease - Confusion Matrix")

plt.tight_layout()

confusion_matrix_path = (
    OUTPUT_PATH / "confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved to:")
print(confusion_matrix_path)


# ============================================================
# FINISH
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION ANALYSIS COMPLETED!")
print("=" * 60)