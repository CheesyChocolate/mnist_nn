import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from src.modules.data_loader import MNISTDataLoader
from src.modules.models import CNNModel, MLPModel, SVMModel, RandomForestModel, KNNModel
from src.modules.metrics import ModelEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='MNIST Pattern Recognition')
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'mlp', 'svm', 'rf', 'knn', 'all'],
                        help='Model to train (default: cnn)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for training (default: 10)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training (default: 128)')
    parser.add_argument('--no-reshape', action='store_true',
                        help='Do not reshape data for CNN (default: False)')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='Sample size for training (default: None - use all data)')
    return parser.parse_args()


def run_cnn(x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """Run CNN model on MNIST data."""
    print("Training CNN model...")
    
    # Initialize model
    model = CNNModel()
    model.build_model()
    
    # Train model
    history = model.train(x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)
    
    # Evaluate model
    evaluator = ModelEvaluator("CNN")
    metrics = evaluator.evaluate_model(model.model, x_test, y_test)
    evaluator.save_metrics(metrics)
    evaluator.plot_training_history(history)
    evaluator.plot_sample_predictions(model.model, x_test, y_test)
    
    print(f"CNN Accuracy: {metrics['accuracy']:.4f}")
    return model, metrics


def run_mlp(x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """Run MLP model on MNIST data."""
    print("Training MLP model...")
    
    # Initialize model
    model = MLPModel()
    model.build_model()
    
    # Train model
    history = model.train(x_train, y_train, x_test, y_test, batch_size=batch_size, epochs=epochs)
    
    # Evaluate model
    evaluator = ModelEvaluator("MLP")
    metrics = evaluator.evaluate_model(model.model, x_test, y_test)
    evaluator.save_metrics(metrics)
    evaluator.plot_training_history(history)
    evaluator.plot_sample_predictions(model.model, x_test, y_test)
    
    print(f"MLP Accuracy: {metrics['accuracy']:.4f}")
    return model, metrics


def run_svm(x_train, y_train, x_test, y_test, sample_size=None):
    """Run SVM model on MNIST data."""
    print("Training SVM model...")
    
    # If sample_size is provided, use a subset of the data for faster training
    if sample_size is not None:
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        x_train_sample = x_train[indices]
        y_train_sample = y_train[indices]
    else:
        x_train_sample = x_train
        y_train_sample = y_train
    
    # Initialize model
    model = SVMModel()
    
    # Train model
    model.train(x_train_sample, y_train_sample)
    
    # Evaluate model
    evaluator = ModelEvaluator("SVM")
    metrics = evaluator.evaluate_model(model, x_test, y_test)
    evaluator.save_metrics(metrics)
    evaluator.plot_sample_predictions(model, x_test, y_test)
    
    print(f"SVM Accuracy: {metrics['accuracy']:.4f}")
    return model, metrics


def run_random_forest(x_train, y_train, x_test, y_test, sample_size=None):
    """Run Random Forest model on MNIST data."""
    print("Training Random Forest model...")
    
    # If sample_size is provided, use a subset of the data for faster training
    if sample_size is not None:
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        x_train_sample = x_train[indices]
        y_train_sample = y_train[indices]
    else:
        x_train_sample = x_train
        y_train_sample = y_train
    
    # Initialize model
    model = RandomForestModel()
    
    # Train model
    model.train(x_train_sample, y_train_sample)
    
    # Evaluate model
    evaluator = ModelEvaluator("RandomForest")
    metrics = evaluator.evaluate_model(model, x_test, y_test)
    evaluator.save_metrics(metrics)
    evaluator.plot_sample_predictions(model, x_test, y_test)
    
    print(f"Random Forest Accuracy: {metrics['accuracy']:.4f}")
    return model, metrics


def run_knn(x_train, y_train, x_test, y_test, sample_size=None):
    """Run KNN model on MNIST data."""
    print("Training KNN model...")
    
    # If sample_size is provided, use a subset of the data for faster training
    if sample_size is not None:
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        x_train_sample = x_train[indices]
        y_train_sample = y_train[indices]
    else:
        x_train_sample = x_train
        y_train_sample = y_train
    
    # Initialize model
    model = KNNModel()
    
    # Train model
    model.train(x_train_sample, y_train_sample)
    
    # Evaluate model
    evaluator = ModelEvaluator("KNN")
    metrics = evaluator.evaluate_model(model, x_test, y_test)
    evaluator.save_metrics(metrics)
    evaluator.plot_sample_predictions(model, x_test, y_test)
    
    print(f"KNN Accuracy: {metrics['accuracy']:.4f}")
    return model, metrics


def main():
    """Main function to run the workflow."""
    # Parse arguments
    args = parse_args()
    
    # Load data
    data_loader = MNISTDataLoader(reshape=not args.no_reshape)
    (x_train, y_train), (x_test, y_test) = data_loader.load_data()
    
    print("MNIST data loaded:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Dictionary to store results
    results = {}
    
    # Run selected model(s)
    if args.model == 'cnn' or args.model == 'all':
        cnn_model, cnn_metrics = run_cnn(x_train, y_train, x_test, y_test, 
                                         epochs=args.epochs, batch_size=args.batch_size)
        results['cnn'] = cnn_metrics
    
    if args.model == 'mlp' or args.model == 'all':
        mlp_model, mlp_metrics = run_mlp(x_train, y_train, x_test, y_test, 
                                         epochs=args.epochs, batch_size=args.batch_size)
        results['mlp'] = mlp_metrics
    
    if args.model == 'svm' or args.model == 'all':
        svm_model, svm_metrics = run_svm(x_train, y_train, x_test, y_test, 
                                         sample_size=args.sample_size)
        results['svm'] = svm_metrics
    
    if args.model == 'rf' or args.model == 'all':
        rf_model, rf_metrics = run_random_forest(x_train, y_train, x_test, y_test, 
                                                sample_size=args.sample_size)
        results['rf'] = rf_metrics
    
    if args.model == 'knn' or args.model == 'all':
        knn_model, knn_metrics = run_knn(x_train, y_train, x_test, y_test, 
                                        sample_size=args.sample_size)
        results['knn'] = knn_metrics
    
    # Compare results if multiple models were run
    if args.model == 'all':
        compare_models(results)


def compare_models(results):
    """Compare accuracy of different models."""
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    
    # Create output directory if it doesn't exist
    os.makedirs('../../doc/fig', exist_ok=True)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies)
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.savefig('../../doc/fig/model_comparison.png')
    plt.close()
    
    # Save comparison to file
    with open('../../doc/out/model_comparison.txt', 'w') as f:
        f.write("Model Accuracy Comparison\n")
        f.write("========================\n\n")
        for model in models:
            f.write(f"{model.upper()}: {results[model]['accuracy']:.4f}\n")


if __name__ == "__main__":
    main()
