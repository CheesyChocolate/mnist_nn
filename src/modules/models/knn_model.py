import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    """K-Nearest Neighbors model for MNIST classification."""
    
    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Initialize the KNN model.
        
        Args:
            n_neighbors: Number of neighbors
            weights: Weight function used in prediction
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            n_jobs=-1
        )
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the KNN model.
        
        Args:
            x_train: Training data (flattened)
            y_train: Training labels (not one-hot encoded)
            x_val: Validation data (not used for KNN)
            y_val: Validation labels (not used for KNN)
            
        Returns:
            Trained model
        """
        # Flatten the input if it's not already flat
        if len(x_train.shape) > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)
        
        # Convert one-hot encoded labels if needed
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            y_train = y_train.argmax(axis=1)
        
        # Train the model
        self.model.fit(x_train, y_train)
        
        return self.model
    
    def predict(self, x):
        """Predict using the KNN model."""
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        return self.model.predict(x)
    
    def predict_proba(self, x):
        """Get probability estimates for test data."""
        # Flatten the input if it's not already flat
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        return self.model.predict_proba(x) 