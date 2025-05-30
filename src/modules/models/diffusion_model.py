"""Diffusion model for pattern recognition in limited data scenarios."""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

class DiffusionModel:
    """Conditional Diffusion model for pattern recognition with semi-supervised learning."""

    def __init__(self, input_shape=(28, 28, 1), num_classes=10, diffusion_steps=1000, beta_schedule='linear'):
        """
        Initialize the Diffusion model for pattern recognition.
        
        Args:
            input_shape: The shape of input images (height, width, channels)
            num_classes: Number of classes for classification
            diffusion_steps: Number of steps in the diffusion process
            beta_schedule: Schedule for noise level (linear or cosine)
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.diffusion_steps = diffusion_steps
        
        # Set up noise schedule
        if beta_schedule == 'linear':
            self.beta = np.linspace(1e-4, 0.02, diffusion_steps)
        elif beta_schedule == 'cosine':
            steps = diffusion_steps + 1
            x = np.linspace(0, steps, steps)
            alphas_cumprod = np.cos(((x / steps) + 0.008) / 1.008 * np.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = np.clip(betas, 0.0001, 0.9999)
        
        self.alpha = 1. - self.beta
        self.alpha_bar = np.cumprod(self.alpha)
        
        # Create model components
        self.model = None
        self.classifier = None
        self.build_model()
    
    def build_denoiser_network(self):
        """Build the U-Net style denoiser network for diffusion."""
        # Input for image
        inputs = layers.Input(shape=self.input_shape)
        
        # Input for timestep embedding
        time_input = layers.Input(shape=(1,))
        time_embedding = layers.Embedding(self.diffusion_steps, 32)(time_input)
        time_embedding = layers.Dense(self.input_shape[0] * self.input_shape[1])(time_embedding)
        time_embedding = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(time_embedding)
        
        # Input for class conditioning
        class_input = layers.Input(shape=(1,))
        class_embedding = layers.Embedding(self.num_classes, 32)(class_input)
        class_embedding = layers.Dense(self.input_shape[0] * self.input_shape[1])(class_embedding)
        class_embedding = layers.Reshape((self.input_shape[0], self.input_shape[1], 1))(class_embedding)
        
        # Combine inputs
        x = layers.Concatenate()([inputs, time_embedding, class_embedding])
        
        # Encoder
        skips = []
        
        # Down 1
        x = layers.Conv2D(32, 3, padding='same', activation='swish')(x)
        x = layers.Conv2D(32, 3, padding='same', activation='swish')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
        
        # Down 2
        x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        skips.append(x)
        x = layers.MaxPooling2D()(x)
        
        # Bottleneck
        x = layers.Conv2D(128, 3, padding='same', activation='swish')(x)
        x = layers.Conv2D(128, 3, padding='same', activation='swish')(x)
        
        # Up 1
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skips.pop()])
        x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        x = layers.Conv2D(64, 3, padding='same', activation='swish')(x)
        
        # Up 2
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skips.pop()])
        x = layers.Conv2D(32, 3, padding='same', activation='swish')(x)
        x = layers.Conv2D(32, 3, padding='same', activation='swish')(x)
        
        # Output
        outputs = layers.Conv2D(1, 3, padding='same')(x)
        
        return models.Model([inputs, time_input, class_input], outputs)
    
    def build_classifier(self):
        """Build a classifier network that takes noisy images as input."""
        inputs = layers.Input(shape=self.input_shape)
        
        # Simple CNN classifier
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        return models.Model(inputs, outputs)
    
    def build_model(self):
        """Build and compile models needed for diffusion-based pattern recognition."""
        print("Building diffusion model...")
        
        # Build the denoiser model
        self.model = self.build_denoiser_network()
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='mse'
        )
        
        # Build the classifier
        self.classifier = self.build_classifier()
        self.classifier.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def q_sample(self, x_0, t):
        """Forward diffusion process: add noise to the input image."""
        # Extract alphas for the given timestep
        a_bar = self.alpha_bar[t]
        
        # Get noise
        eps = np.random.normal(size=x_0.shape)
        
        # Return noisy image: x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps
        return np.sqrt(a_bar) * x_0 + np.sqrt(1 - a_bar) * eps, eps
    
    def generate_samples(self, num_samples=10, class_labels=None, save_path=None):
        """Generate samples using the trained diffusion model."""
        if class_labels is None:
            # Generate one sample for each class
            class_labels = np.arange(min(num_samples, self.num_classes))
            
        # Start from random noise
        x_T = np.random.normal(size=(len(class_labels), *self.input_shape))
        
        # Iteratively denoise
        for t in range(self.diffusion_steps - 1, -1, -1):
            # Create batch of same timestep
            timesteps = np.full(len(class_labels), t)
            
            # Get model prediction for noise
            predicted_noise = self.model.predict(
                [x_T, timesteps, class_labels], verbose=0
            )
            
            # Compute denoised image
            alpha_t = self.alpha[t]
            alpha_bar_t = self.alpha_bar[t]
            
            if t > 0:
                # Add noise for t > 0
                z = np.random.normal(size=x_T.shape) if t > 0 else 0
                beta_t = self.beta[t]
                x_T = (1 / np.sqrt(alpha_t)) * (
                    x_T - (beta_t / np.sqrt(1 - alpha_bar_t)) * predicted_noise
                ) + np.sqrt(beta_t) * z
            else:
                # For t=0, just get the predicted image
                x_T = (1 / np.sqrt(alpha_t)) * (
                    x_T - (self.beta[t] / np.sqrt(1 - alpha_bar_t)) * predicted_noise
                )
        
        # Clip the images to [0, 1]
        x_T = np.clip(x_T, 0, 1)
        
        # Save if path is provided
        if save_path:
            fig, axs = plt.subplots(1, len(x_T), figsize=(2 * len(x_T), 2))
            for i, (img, label) in enumerate(zip(x_T, class_labels)):
                axs[i].imshow(img.reshape(self.input_shape[0], self.input_shape[1]), cmap='gray')
                axs[i].set_title(f'Class: {label}')
                axs[i].axis('off')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
        
        return x_T
    
    def train_with_limited_data(self, X_labeled, y_labeled, X_unlabeled, X_test, y_test, 
                                diffusion_epochs=50, classifier_epochs=10, batch_size=32):
        """
        Train the diffusion model with limited labeled data and unlabeled data.
        
        Args:
            X_labeled: Labeled training images
            y_labeled: Labels for the labeled training images
            X_unlabeled: Unlabeled training images
            X_test: Test images
            y_test: Test labels
            diffusion_epochs: Number of epochs for diffusion model training
            classifier_epochs: Number of epochs for classifier training
            batch_size: Batch size for training
        """
        # Convert labels to one-hot encoding for the classifier
        y_labeled_cat = to_categorical(y_labeled, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        # Step 1: Train the diffusion model on all available data (labeled + unlabeled)
        print("Training diffusion model with combined labeled and unlabeled data...")
        
        # Combine labeled and unlabeled data for diffusion training
        X_combined = np.concatenate([X_labeled, X_unlabeled], axis=0)
        
        # Create classifier pseudo-labels for unlabeled data
        if len(X_unlabeled) > 0:
            # Initially train classifier on labeled data
            self.classifier.fit(
                X_labeled, y_labeled_cat,
                epochs=5,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
            
            # Generate pseudo-labels for unlabeled data
            pseudo_labels = np.argmax(self.classifier.predict(X_unlabeled, verbose=0), axis=1)
        else:
            pseudo_labels = []
        
        # Combine real labels and pseudo-labels
        all_labels = np.concatenate([y_labeled, pseudo_labels], axis=0)
        
        # Training loop for the diffusion model
        losses = []
        for epoch in range(diffusion_epochs):
            epoch_losses = []
            for i in range(0, len(X_combined), batch_size):
                # Get batch
                batch_images = X_combined[i:i+batch_size]
                batch_labels = all_labels[i:i+batch_size]
                batch_size_actual = len(batch_images)
                
                # Sample random timesteps
                timesteps = np.random.randint(0, self.diffusion_steps, size=batch_size_actual)
                
                # Add noise to images according to timesteps
                noisy_images = []
                target_noise = []
                for j, img in enumerate(batch_images):
                    noisy_img, target_eps = self.q_sample(img, timesteps[j])
                    noisy_images.append(noisy_img)
                    target_noise.append(target_eps)
                
                noisy_images = np.array(noisy_images)
                target_noise = np.array(target_noise)
                
                # Train the model to predict the noise
                loss = self.model.train_on_batch(
                    [noisy_images, timesteps, batch_labels],
                    target_noise
                )
                epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0 or epoch == diffusion_epochs - 1:
                print(f"Epoch {epoch+1}/{diffusion_epochs} - Diffusion loss: {avg_loss:.4f}")
                
                # Generate samples for visualization
                self.generate_samples(
                    num_samples=10,
                    save_path=f"doc/fig/diffusion_samples_epoch_{epoch+1}.png"
                )
        
        # Plot diffusion training curve
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Diffusion Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.savefig("doc/fig/diffusion_training_loss.png")
        plt.close()
        
        # Step 2: Generate augmented images for classifier training
        print("Generating augmented images for classifier training...")
        augmented_images = []
        augmented_labels = []
        
        # Generate additional samples for each class
        samples_per_class = 100  # Number of additional samples per class
        for class_idx in range(self.num_classes):
            # Count existing samples for this class
            class_count = np.sum(y_labeled == class_idx)
            
            # Generate more samples if needed
            if class_count < samples_per_class:
                num_to_generate = samples_per_class - class_count
                class_labels = np.full(num_to_generate, class_idx)
                
                generated_samples = self.generate_samples(
                    num_samples=num_to_generate,
                    class_labels=class_labels
                )
                
                augmented_images.append(generated_samples)
                augmented_labels.append(class_labels)
        
        # Combine original and augmented data if any were generated
        if augmented_images:
            augmented_images = np.concatenate(augmented_images, axis=0)
            augmented_labels = np.concatenate(augmented_labels, axis=0)
            
            X_train_aug = np.concatenate([X_labeled, augmented_images], axis=0)
            y_train_aug = np.concatenate([y_labeled, augmented_labels], axis=0)
            y_train_aug_cat = to_categorical(y_train_aug, self.num_classes)
        else:
            X_train_aug = X_labeled
            y_train_aug_cat = y_labeled_cat
        
        # Step 3: Train the classifier with augmented data
        print("Training classifier with augmented data...")
        history = self.classifier.fit(
            X_train_aug, y_train_aug_cat,
            validation_data=(X_test, y_test_cat),
            epochs=classifier_epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Plot classifier training curve
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Classifier Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'])
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Classifier Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig("doc/fig/diffusion_classifier_training.png")
        plt.close()
        
        # Generate final samples
        self.generate_samples(
            num_samples=10,
            save_path="doc/fig/diffusion_final_samples.png"
        )
        
        print("Diffusion model training completed")
    
    def predict(self, X):
        """Predict class labels for the input images."""
        return np.argmax(self.classifier.predict(X, verbose=0), axis=1) 