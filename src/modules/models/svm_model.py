import numpy as np
from sklearn.svm import SVC

class SVMModel:
    """Support Vector Machine model for MNIST classification."""
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize the SVM model.
        
        Args:
            kernel: Kernel type to be used in the algorithm
            C: Regularization parameter
            gamma: Kernel coefficient
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the SVM model.
        
        Args:
            x_train: Training data (flattened)
            y_train: Training labels (not one-hot encoded)
            x_val: Validation data (not used for SVM)
            y_val: Validation labels (not used for SVM)
            
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
        """Predict using the SVM model."""
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