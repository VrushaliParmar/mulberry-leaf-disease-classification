import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "outputs" / "processed_dataset" / "train"
VALIDATION_PATH = PROJECT_ROOT / "outputs" / "processed_dataset" / "validation"

SAVED_MODEL_PATH = PROJECT_ROOT / "saved_models"
PLOTS_PATH = PROJECT_ROOT / "outputs" / "plots"

SAVED_MODEL_PATH.mkdir(parents=True, exist_ok=True)
PLOTS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
EPOCHS = 20


# ============================================================
# 3. LOAD TRAINING DATASET
# ============================================================

print("=" * 60)
print("MULBERRY LEAF DISEASE - MODEL TRAINING")
print("=" * 60)

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)


# ============================================================
# 4. LOAD VALIDATION DATASET
# ============================================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# 5. CLASS NAMES
# ============================================================

class_names = train_dataset.class_names

print("\n" + "=" * 60)
print("CLASSES")
print("=" * 60)

for index, class_name in enumerate(class_names):
    print(f"{index} --> {class_name}")

NUM_CLASSES = len(class_names)

print(f"\nNumber of classes: {NUM_CLASSES}")


# ============================================================
# 6. COUNT TRAINING SAMPLES
# ============================================================

train_labels = np.concatenate(
    [labels.numpy() for _, labels in train_dataset],
    axis=0
)

print("\n" + "=" * 60)
print("TRAINING DATA")
print("=" * 60)

for class_index, class_name in enumerate(class_names):
    count = np.sum(train_labels == class_index)
    print(f"{class_name:<25}: {count}")

print(f"{'Total':<25}: {len(train_labels)}")


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=train_labels
)

class_weights = {
    index: weight
    for index, weight in enumerate(class_weights_array)
}

print("\n" + "=" * 60)
print("CLASS WEIGHTS")
print("=" * 60)

for class_index, weight in class_weights.items():
    print(f"{class_names[class_index]:<25}: {weight:.4f}")


# ============================================================
# 8. IMPROVE DATA PIPELINE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# 9. DATA AUGMENTATION
# ============================================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.10),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomContrast(0.20),
], name="data_augmentation")


# ============================================================
# 10. LOAD MOBILENETV2
# ============================================================

print("\n" + "=" * 60)
print("LOADING MOBILENETV2")
print("=" * 60)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

print("MobileNetV2 loaded successfully!")
print("Pretrained layers frozen.")


# ============================================================
# 11. BUILD MODEL
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="input_image"
)

# Data augmentation
x = data_augmentation(inputs)

# MobileNetV2 preprocessing
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

# Feature extraction
x = base_model(
    x,
    training=False
)

# Convert feature maps into a single feature vector
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Reduce overfitting
x = tf.keras.layers.Dropout(0.30)(x)

# Classification layer
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="disease_prediction"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="mulberry_disease_classifier"
)


# ============================================================
# 12. COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 13. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

best_model_path = SAVED_MODEL_PATH / "best_model_v2.keras"

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath=best_model_path,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ============================================================
# 15. START TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Image size : {IMAGE_SIZE}")
print("=" * 60)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)


# ============================================================
# 16. SAVE FINAL MODEL
# ============================================================

final_model_path = SAVED_MODEL_PATH / "final_model_v2.keras"

model.save(final_model_path)

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nBest model saved at:")
print(best_model_path)

print("\nFinal model saved at:")
print(final_model_path)


# ============================================================
# 17. TRAINING ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("V2 Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

accuracy_path = PLOTS_PATH / "accuracy_v2.png"

plt.savefig(
    accuracy_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nAccuracy graph saved at:")
print(accuracy_path)


# ============================================================
# 18. TRAINING LOSS GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("V2 Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

loss_path = PLOTS_PATH / "loss_v2.png"

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print("\nLoss graph saved at:")
print(loss_path)


# ============================================================
# 19. FINAL TRAINING RESULTS
# ============================================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

best_train_accuracy = max(
    history.history["accuracy"]
)

print("\n" + "=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

print(
    f"Best Training Accuracy   : "
    f"{best_train_accuracy * 100:.2f}%"
)

print(
    f"Best Validation Accuracy : "
    f"{best_val_accuracy * 100:.2f}%"
)

print("=" * 60)

print("\nV2 MODEL TRAINING SUCCESSFUL!")