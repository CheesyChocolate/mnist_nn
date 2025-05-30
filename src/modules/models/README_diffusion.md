# Diffusion Model for Semi-Supervised Learning

This module implements a conditional diffusion model for pattern recognition with limited labeled data, focusing on the MNIST dataset.

## Overview

Diffusion models are a class of generative models that learn to gradually denoise a signal that has been corrupted by a forward diffusion process. This implementation adapts diffusion models for semi-supervised learning, where we have access to a large amount of unlabeled data but only a small amount of labeled data.

## Approach

The key insight of this approach is to leverage the powerful generative capabilities of diffusion models to learn meaningful representations from both labeled and unlabeled data. The model consists of two main components:

1. **Conditional Denoising Diffusion Model**: A U-Net style network that learns to denoise images conditioned on both timestep and class label.

2. **Classifier Network**: A CNN-based classifier that uses the learned representations from the diffusion model.

## Training Process

The training process involves several stages:

1. **Initial Classifier Training**: The classifier is first trained on the limited labeled data to generate pseudo-labels for unlabeled data.

2. **Diffusion Model Training**: The diffusion model is trained on both labeled data (with true labels) and unlabeled data (with pseudo-labels) to learn the denoising process.

3. **Data Augmentation**: After training, the diffusion model generates additional synthetic examples for underrepresented classes.

4. **Final Classifier Training**: The classifier is trained on the combined dataset of real labeled data and generated synthetic data.

## Implementation Details

### Diffusion Process

- **Forward Diffusion**: Gradually adds Gaussian noise to images according to a predefined schedule.
- **Reverse Diffusion**: Learns to gradually remove noise starting from pure noise to generate clean images.

### Model Architecture

- **Denoiser Network**: U-Net style architecture with skip connections.
- **Time Embedding**: Embeds the diffusion timestep to condition the denoising process.
- **Class Conditioning**: Incorporates class information to enable conditional generation.
- **Classifier**: Simple CNN architecture optimized for MNIST classification.

## Usage Example

```python
from src.modules.models.diffusion_model import DiffusionModel

# Initialize the model
model = DiffusionModel(input_shape=(28, 28, 1), num_classes=10, diffusion_steps=100)

# Train with limited labeled data
model.train_with_limited_data(
    X_labeled,
    y_labeled,
    X_unlabeled,
    X_test,
    y_test,
    diffusion_epochs=20,
    classifier_epochs=10,
    batch_size=32
)

# Generate samples for a specific class
samples = model.generate_samples(num_samples=5, class_labels=[3, 3, 3, 3, 3])

# Make predictions
predictions = model.predict(X_test)
```

## Results

The diffusion-based semi-supervised approach significantly outperforms standard supervised learning when labeled data is limited. It demonstrates how generative models can be effectively leveraged for classification tasks in scenarios with scarce labeled data.

This approach is particularly valuable in domains where obtaining labeled data is expensive or time-consuming, such as medical imaging, industrial defect detection, or specialized document analysis. 