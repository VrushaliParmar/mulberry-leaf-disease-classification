import tensorflow as tf
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_model_v2.keras"

TEST_PATH = PROJECT_ROOT / "outputs" / "processed_dataset" / "test"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# START
# ============================================================

print("=" * 60)
print("MULBERRY LEAF DISEASE - V2 TEST EVALUATION")
print("=" * 60)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best V2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading TEST dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED
)

print("\nTest dataset loaded successfully!")


# ============================================================
# CLASS NAMES
# ============================================================

class_names = test_dataset.class_names

print("\n" + "=" * 60)
print("CLASSES")
print("=" * 60)

for index, class_name in enumerate(class_names):
    print(f"{index} --> {class_name}")


# ============================================================
# TEST DATA COUNT
# ============================================================

total_test_images = 0

for _, labels in test_dataset:
    total_test_images += len(labels)

print("\nTotal test images:", total_test_images)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL ON UNSEEN TEST DATA")
print("=" * 60)

loss, accuracy = model.evaluate(
    test_dataset,
    verbose=1
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(f"Test Loss       : {loss:.4f}")
print(f"Test Accuracy   : {accuracy:.4f}")
print(f"Test Accuracy   : {accuracy * 100:.2f}%")

print("=" * 60)

print("\nV2 TEST EVALUATION COMPLETED!")