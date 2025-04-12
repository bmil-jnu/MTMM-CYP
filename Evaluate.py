import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    average_precision_score, matthews_corrcoef,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import defaultdict


def safe_division(numerator, denominator):
    """Perform division safely, avoiding division-by-zero."""
    return numerator / denominator if denominator else 0


def print_metrics(metrics):
    """Print formatted evaluation metrics."""
    metric_names = ['Accuracy', 'AUC-ROC', 'AUC-PR', 'MCC', 'Sensitivity/Recall', 'Specificity', 'Precision',
                    'F1-score']
    for name, value in zip(metric_names, metrics):
        if value is None:
            print(f'{name}: N/A')
        else:
            print(f'{name}: {value:.4f}')


def plot_curve(x, y, xlabel, ylabel, title, legend_text):
    """Plot ROC or Precision-Recall curves."""
    plt.figure()
    plt.plot(x, y, color='darkorange', lw=2, label=legend_text)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def printPerformance(labels, probs, threshold=0.40, printout=True):
    """
    Evaluate and print classification performance.

    Parameters:
        labels (array-like): True binary labels.
        probs (array-like): Predicted probabilities.
        threshold (float): Probability threshold for binary classification.
        printout (bool): Whether to print metrics and plot curves.
    """
    labels = np.array(labels)
    probs = np.array(probs)

    # Filter out invalid labels (-1)
    valid_indices = labels != -1
    labels = labels[valid_indices]
    probs = probs[valid_indices]

    # Binary predictions based on threshold
    predicted_labels = (probs >= threshold).astype(int)

    # Compute confusion matrix components
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()

    # Compute evaluation metrics
    sensitivity = safe_division(tp, tp + fn)
    specificity = safe_division(tn, tn + fp)
    precision = safe_division(tp, tp + fp)
    f1_score = safe_division(2 * precision * sensitivity, precision + sensitivity)

    metrics = [
        accuracy_score(labels, predicted_labels),
        roc_auc_score(labels, probs),
        average_precision_score(labels, probs),
        matthews_corrcoef(labels, predicted_labels),
        sensitivity,
        specificity,
        precision,
        f1_score
    ]

    # Print metrics
    if printout:
        print_metrics(metrics)

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(labels, probs)
    plot_curve(fpr, tpr, 'False Positive Rate', 'True Positive Rate', 'Receiver Operating Characteristic',
               f'ROC curve (AUC = {metrics[1]:.3f})')

    # Plot Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(labels, probs)
    plot_curve(recall_curve, precision_curve, 'Recall', 'Precision', 'Precision-Recall Curve',
               f'PR curve (AUC = {metrics[2]:.3f})')

    return metrics
