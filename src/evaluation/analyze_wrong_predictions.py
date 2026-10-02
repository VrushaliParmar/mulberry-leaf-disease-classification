import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "best_model_v2.keras"
)

TEST_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "processed_dataset"
    / "test"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
    / "wrong_predictions_v2"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("V2 WRONG PREDICTION ANALYSIS")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_dataset.class_names

print("\nClasses:")

for index, name in enumerate(class_names):
    print(f"{index} --> {name}")


# ============================================================
# GET IMAGES AND LABELS
# ============================================================

images = []
true_labels = []

for batch_images, batch_labels in test_dataset:

    images.extend(
        batch_images.numpy()
    )

    true_labels.extend(
        batch_labels.numpy()
    )

images = np.array(images)
true_labels = np.array(true_labels)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    images,
    batch_size=BATCH_SIZE,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

confidence_scores = np.max(
    predictions,
    axis=1
)


# ============================================================
# FIND WRONG PREDICTIONS
# ============================================================

wrong_indices = np.where(
    predicted_labels != true_labels
)[0]


print("\n" + "=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

print(
    f"Total incorrect predictions: "
    f"{len(wrong_indices)}"
)

print(
    f"Total correct predictions: "
    f"{len(images) - len(wrong_indices)}"
)


# ============================================================
# PRINT WRONG PREDICTIONS
# ============================================================

for number, index in enumerate(wrong_indices):

    actual_class = class_names[
        true_labels[index]
    ]

    predicted_class = class_names[
        predicted_labels[index]
    ]

    confidence = (
        confidence_scores[index] * 100
    )

    print("\n" + "-" * 60)

    print(
        f"Wrong Prediction #{number + 1}"
    )

    print(
        f"Actual      : {actual_class}"
    )

    print(
        f"Predicted   : {predicted_class}"
    )

    print(
        f"Confidence  : {confidence:.2f}%"
    )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

if len(wrong_indices) > 0:

    number_to_show = min(
        len(wrong_indices),
        11
    )

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(16, 12)
    )

    axes = axes.flatten()

    for plot_index in range(
        len(axes)
    ):

        if plot_index >= number_to_show:

            axes[plot_index].axis(
                "off"
            )

            continue

        image_index = wrong_indices[
            plot_index
        ]

        axes[plot_index].imshow(
            images[image_index].astype(
                "uint8"
            )
        )

        actual = class_names[
            true_labels[image_index]
        ]

        predicted = class_names[
            predicted_labels[image_index]
        ]

        confidence = (
            confidence_scores[
                image_index
            ] * 100
        )

        axes[plot_index].set_title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Confidence: {confidence:.1f}%"
        )

        axes[plot_index].axis(
            "off"
        )

    plt.suptitle(
        "V2 Model - Incorrect Test Predictions",
        fontsize=16
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_PATH
        / "wrong_predictions_v2.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n" + "=" * 60)

    print(
        "Wrong prediction visualization saved to:"
    )

    print(output_file)

else:

    print(
        "\nNo incorrect predictions found!"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("WRONG PREDICTION ANALYSIS COMPLETED!")
print("=" * 60)