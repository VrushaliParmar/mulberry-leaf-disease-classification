import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "best_model_v2.keras"
)

TEST_DATASET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset"
    / "test"
)

PLOTS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
)

PLOTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# START
# ============================================================

print("=" * 60)
print("MULBERRY LEAF DISEASE - V2 DETAILED TEST EVALUATION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V2 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading TEST dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest dataset loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = test_dataset.class_names

print("\nClasses:")

for index, class_name in enumerate(class_names):
    print(f"{index} --> {class_name}")


# ============================================================
# COUNT TEST IMAGES
# ============================================================

total_test_images = 0

for _, labels in test_dataset:
    total_test_images += len(labels)

print("\nTotal test images:", total_test_images)


# ============================================================
# BASIC MODEL EVALUATION
# ============================================================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


print("\n" + "=" * 60)
print("V2 TEST RESULTS")
print("=" * 60)

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")

print("=" * 60)


# ============================================================
# GET TRUE LABELS
# ============================================================

print("\nGenerating predictions...")

true_labels = []

for _, labels in test_dataset:
    true_labels.extend(
        labels.numpy()
    )

true_labels = np.array(true_labels)


# ============================================================
# GET PREDICTIONS
# ============================================================

predictions = model.predict(
    test_dataset,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

report = classification_report(
    true_labels,
    predicted_labels,
    target_names=class_names,
    zero_division=0
)

print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

cm = confusion_matrix(
    true_labels,
    predicted_labels
)

print(cm)


# ============================================================
# DISPLAY CONFUSION MATRIX
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(
    figsize=(8, 7)
)

disp.plot(
    ax=ax,
    values_format="d",
    xticks_rotation=45
)

plt.title(
    "V2 Model - Mulberry Leaf Disease Test Confusion Matrix"
)

plt.tight_layout()


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_matrix_path = (
    PLOTS_PATH
    / "confusion_matrix_v2_test.png"
)

plt.savefig(
    confusion_matrix_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nConfusion matrix saved to:")

print(confusion_matrix_path)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS ANALYSIS")
print("=" * 60)

for index, class_name in enumerate(class_names):

    class_total = np.sum(
        true_labels == index
    )

    class_correct = np.sum(
        (true_labels == index)
        &
        (predicted_labels == index)
    )

    if class_total > 0:
        class_accuracy = (
            class_correct / class_total
        ) * 100
    else:
        class_accuracy = 0

    print(
        f"{class_name:<25} : "
        f"{class_accuracy:.2f}% "
        f"({class_correct}/{class_total})"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("V2 DETAILED TEST EVALUATION COMPLETED!")
print("=" * 60)