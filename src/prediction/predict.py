import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "saved_models"
    / "best_model_v2.keras"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "Disease Free leaves",
    "Leaf Rust",
    "Leaf spot"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("MULBERRY LEAF DISEASE PREDICTION")
print("=" * 60)

print("\nLoading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

print(f"\nModel path:")
print(MODEL_PATH)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_image(image_path):

    print("\nLoading image...")
    print(f"Image: {image_path}")

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(image_path).convert("RGB")

    print(f"Original image size: {image.size}")

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    image = image.resize(IMAGE_SIZE)

    # --------------------------------------------------------
    # Convert to NumPy array
    # --------------------------------------------------------

    image_array = np.array(image)

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = predictions[0]

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index] * 100

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"\nPredicted Disease : {predicted_class}")
    print(f"Confidence        : {confidence:.2f}%")

    print("\nClass Probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):

        probability = probabilities[index] * 100

        print(
            f"{class_name:<25}: "
            f"{probability:.2f}%"
        )

    print("=" * 60)

    return predicted_class, confidence


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    image_path = input(
        "\nEnter path of leaf image: "
    ).strip()

    image_path = Path(image_path)

    if not image_path.exists():

        print("\nERROR:")
        print("Image file does not exist.")

    else:

        predict_image(image_path)