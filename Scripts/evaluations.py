
# ... accuracy, precision, recall and f1-score metrics which will be
# used for model evaluation as it ensures a comprehensive and reliable assessment for
# classification models.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(test_labels, predictions):
    # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')
        cm = confusion_matrix(test_labels, predictions)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
        


def perform_kfold(model, features, labels, k=5, model_type="sklearn"):
    """
    Perform K-Fold Cross-Validation for a given model.
    :param model: The model instance (SVM, Logistic Regression, or Neural Network).
    :param features: Feature dataset.
    :param labels: Target labels.
    :param k: Number of folds for K-Fold.
    :param model_type: Type of the model ('sklearn' or 'pytorch').
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(features)):
        print(f"\nFold {fold + 1}/{k}")

        # Split data into train and test for the fold
        train_features, test_features = features[train_idx], features[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # Train the model
        model.train(train_features, train_labels)

        # Predict on the test fold
        if model_type == "pytorch":
            predictions = model.predict(test_features).numpy()  # Convert PyTorch tensor to numpy
        else:
            predictions = model.predict(test_features)

        # Evaluate performance for the fold
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted')
        recall = recall_score(test_labels, predictions, average='weighted')
        f1 = f1_score(test_labels, predictions, average='weighted')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        fold_metrics.append([accuracy, precision, recall, f1])

    # Calculate average metrics across all folds
    fold_metrics = np.array(fold_metrics)
    avg_metrics = np.mean(fold_metrics, axis=0)

    print("\nAverage Metrics Across All Folds:")
    print(f"Accuracy: {avg_metrics[0]:.4f}")
    print(f"Precision: {avg_metrics[1]:.4f}")
    print(f"Recall: {avg_metrics[2]:.4f}")
    print(f"F1-Score: {avg_metrics[3]:.4f}")

    return {
        "accuracy": avg_metrics[0],
        "precision": avg_metrics[1],
        "recall": avg_metrics[2],
        "f1_score": avg_metrics[3]
    }

def bias_variance_analysis(model, features_train, labels_train, features_test, labels_test, model_type="sklearn"):
    """
    Perform Bias-Variance Analysis for a given model.
    :param model: The model instance (SVM, Logistic Regression, or Neural Network).
    :param features_train: Training feature set.
    :param labels_train: Training labels.
    :param features_test: Testing feature set.
    :param labels_test: Testing labels.
    :param model_type: Type of model ('sklearn' or 'pytorch').
    """
    train_sizes = np.linspace(0.1, 1.0, 10)  # Use different portions of the training set
    train_errors = []
    test_errors = []

    for train_size in train_sizes:
        # Create subset of the training data
        subset_size = int(train_size * len(features_train))
        X_subset, Y_subset = features_train[:subset_size], labels_train[:subset_size]

        # Train the model
        model.train(X_subset, Y_subset)

        # Predict on training data
        if model_type == "pytorch":
            train_predictions = model.predict(X_subset).numpy()
            test_predictions = model.predict(features_test).numpy()
        else:
            train_predictions = model.predict(X_subset)
            test_predictions = model.predict(features_test)

        # Calculate errors
        train_error = 1 - accuracy_score(Y_subset, train_predictions)
        test_error = 1 - accuracy_score(labels_test, test_predictions)

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plot training and testing errors
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors, label="Training Error", marker='o')
    plt.plot(train_sizes, test_errors, label="Validation Error", marker='s')
    plt.xlabel("Training Set Size")
    plt.ylabel("Error")
    plt.title("Bias-Variance Analysis")
    plt.legend()
    plt.grid(True)
    plt.show()

    print("\nBias-Variance Analysis Complete.")
