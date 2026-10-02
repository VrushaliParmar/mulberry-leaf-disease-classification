# Saved Models

The trained MobileNetV2 model files are not included in this repository.

## Model

- Architecture: MobileNetV2 with ImageNet-pretrained weights
- Input size: 224 × 224 × 3
- Output classes: 3
- Dropout: 0.3
- Optimizer: Adam
- Learning rate: 0.001

## Final Evaluation

- Held-out test set: 165 images
- Test accuracy: 93.33%
- Correct predictions: 154 / 165

The model can be retrained using the training scripts in `src/training/`.
