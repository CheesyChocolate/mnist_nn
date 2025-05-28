import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestModel:
    """Random Forest model for MNIST classification."""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the tree
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the Random Forest model.
        
        Args:
            x_train: Training data (flattened)
            y_train: Training labels (not one-hot encoded)
            x_val: Validation data (not used for RF)
            y_val: Validation labels (not used for RF)
            
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
        """Predict using the Random Forest model."""
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