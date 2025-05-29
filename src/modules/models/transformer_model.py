"""Vision Transformer model for MNIST digit classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them."""

    def __init__(self, img_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor with shape (batch_size, in_channels, img_size, img_size)

        Returns:
            torch.Tensor: Embedded patches with shape (batch_size, n_patches, embed_dim)
        """
        x = self.proj(x)  # (batch_size, embed_dim, n_patches^0.5, n_patches^0.5)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder with multi-head self-attention."""

    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor with shape (batch_size, seq_len, embed_dim)

        Returns:
            torch.Tensor: Output tensor with shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention block
        norm_x = self.norm1(x)
        attn_output, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_output

        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        mlp_ratio=2,
        num_layers=4,  # Reduced from 6 to 4 for better CPU performance
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)

        # Create transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize the position embedding with basic positional encoding
        self._init_pos_embed()

    def _init_pos_embed(self):
        """Initialize positional embedding with simple encoding for faster convergence."""
        n_patches = self.patch_embed.n_patches
        n_tokens = n_patches + 1  # Add 1 for cls token
        embed_dim = self.pos_embed.shape[-1]

        # Create basic position embedding
        pos_embed = torch.zeros(1, n_tokens, embed_dim)
        for i in range(n_tokens):
            for j in range(embed_dim):
                if j % 2 == 0:
                    pos_embed[0, i, j] = np.sin(i / (10000 ** (j / embed_dim)))
                else:
                    pos_embed[0, i, j] = np.cos(i / (10000 ** ((j - 1) / embed_dim)))

        self.pos_embed.data.copy_(pos_embed)

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor with shape (batch_size, in_channels, img_size, img_size)

        Returns:
            torch.Tensor: Class logits with shape (batch_size, num_classes)
        """
        batch_size = x.shape[0]

        # Create patch embeddings
        x = self.patch_embed(x)  # (batch_size, n_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch_size, 1 + n_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take the class token representation
        x = self.norm(x)[:, 0]

        # Classification head
        x = self.head(x)
        return x


class MNISTTransformerModel:
    """MNIST Transformer model wrapper for scikit-learn compatibility."""

    def __init__(
        self,
        img_size=28,
        patch_size=7,
        embed_dim=64,
        num_heads=4,
        num_layers=4,  # Reduced from 6 to 4 for better CPU performance
        batch_size=32,  # Reduced from 64 to 32 for better CPU performance
        num_epochs=5,  # Reduced from 10 to 5 for better CPU performance
        learning_rate=0.001,
        device=None,
    ):
        """Initialize the model.

        Args:
            img_size: Size of the input images
            patch_size: Size of the patches
            embed_dim: Dimension of the embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            batch_size: Batch size for training
            num_epochs: Number of training epochs
            learning_rate: Learning rate for the optimizer
            device: Device to use for training (defaults to GPU if available)
        """
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        # Check for ROCm (AMD GPU) support
        rocm_available = False
        if torch.cuda.is_available():
            if torch.version.hip is not None:  # Check if PyTorch was built with ROCm
                rocm_available = True

        # Determine the device to use
        if device is None:
            if rocm_available:
                print("Using AMD GPU with ROCm")
                self.device = torch.device("cuda")
            elif torch.cuda.is_available():
                print("Using NVIDIA GPU with CUDA")
                self.device = torch.device("cuda")
            else:
                print("Using CPU for training (this may be slow)")
                self.device = torch.device("cpu")
                # For better CPU performance
                torch.set_num_threads(os.cpu_count())
        else:
            self.device = device

        # Initialize the model
        self.model = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(self.device)

        # Initialize optimizer and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y):
        """Train the model.

        Args:
            X: Training data with shape (n_samples, n_features)
            y: Target labels with shape (n_samples,)

        Returns:
            self: The trained model
        """
        # Reshape input to match model expectations
        X = X.reshape(-1, 1, self.img_size, self.img_size)

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.device.type != "cpu",  # Use pin_memory for GPU training
        )

        # Training loop
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in dataloader:
                # Move data to the appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # Print epoch statistics
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = 100 * correct / total
            print(
                f"Epoch {epoch+1}/{self.num_epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%"
            )

        return self

    def predict(self, X):
        """Predict classes for samples in X.

        Args:
            X: Data to predict with shape (n_samples, n_features)

        Returns:
            numpy.ndarray: Predicted class labels
        """
        # Reshape input to match model expectations
        X = X.reshape(-1, 1, self.img_size, self.img_size)

        # Convert numpy array to PyTorch tensor
        X_tensor = torch.FloatTensor(X)

        # Create dataloader for batch processing
        dataloader = DataLoader(
            X_tensor,
            batch_size=self.batch_size,
            pin_memory=self.device.type != "cpu",  # Use pin_memory for GPU training
        )

        # Prediction loop
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for inputs in dataloader:
                # Move data to the appropriate device
                inputs = inputs.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                # Move predictions back to CPU for numpy conversion
                predictions.extend(predicted.cpu().numpy())

        return np.array(predictions)
