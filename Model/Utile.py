import os
import logging
import time
import random
import numpy as np
import math
import datetime
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, mean_squared_error, mean_absolute_error
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from new_pubchemfp import GetPubChemFPs 

# Set random seed for reproducibility
def seed_set(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Create logger for training
def create_logger(output_dir="output", tag="default"):
    log_name = f"training_{tag}_{time.strftime('%Y-%m-%d')}.log"
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = '[%(asctime)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = f'\033[92m[%(asctime)s]\033[0m \033[93m(%(filename)s %(lineno)d):\033[0m \033[95m%(levelname)-5s\033[0m %(message)s'

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger

# Build optimizer based on type
def build_optimizer(model, optimizer_type="adamw", base_lr=0.001, momentum=0.9, weight_decay=1e-4):
    params = model.parameters()
    opt_lower = optimizer_type.lower()

    if opt_lower == 'sgd':
        return torch.optim.SGD(params, lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_lower == 'adam':
        return torch.optim.Adam(params, lr=base_lr, weight_decay=weight_decay)
    elif opt_lower == 'adamw':
        return torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    elif opt_lower == 'ranger':
        try:
            from ranger_adabelief import Ranger
            return Ranger(params, lr=base_lr, weight_decay=weight_decay)
        except ImportError:
            raise ImportError("Ranger optimizer not installed. Run 'pip install ranger-adabelief'.")
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

# Build learning rate scheduler
def build_scheduler(optimizer, scheduler_type="reduce", factor=0.1, patience=10, min_lr=1e-5, steps_per_epoch=None):
    if scheduler_type == "reduce":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr)
    else:
        raise NotImplementedError(f"Unsupported scheduler type: {scheduler_type}")

# Early stopping utility
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0.001, path='checkpoint.pt', trace_func=print, monitor='auc'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.monitor = monitor
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_metric, model):
        score = val_metric if self.monitor == 'auc' else -val_metric
        improvement = score > (self.best_score + self.delta) if self.best_score is not None else True

        if self.best_score is None or improvement:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0
        else:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_metric, model):
        if self.verbose:
            msg = 'AUC' if self.monitor == 'auc' else 'loss'
            self.trace_func(f'Validation {msg} improved. Saving model...')
        torch.save(model.state_dict(), self.path)
        if self.monitor == 'loss':
            self.val_loss_min = val_metric

# Metric functions
def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))

def get_metric_func(metric="auc"):
    if metric == 'auc': return roc_auc_score
    if metric == 'prc': return prc_auc
    if metric == 'rmse': return rmse
    if metric == 'mae': return mean_absolute_error
    raise ValueError(f'Metric "{metric}" not supported.')

def validate_loss_nan(loss, logger, epoch):
    if torch.isnan(loss):
        logger.error(f"NaN loss detected at epoch {epoch}.")
        return True
    return False

# Visualization and performance evaluation utilities
def plot_curve(x, y, xlabel, ylabel, title, legend_text):
    plt.figure()
    plt.plot(x, y, color='darkorange', lw=2, label=legend_text)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def safe_division(numerator, denominator):
    return numerator / denominator if denominator else 0

def print_metrics(metrics):
    names = ['Accuracy', 'AUC-ROC', 'AUC-PR', 'MCC', 'Recall', 'Specificity', 'Precision', 'F1-score']
    for name, value in zip(names, metrics):
        print(f'{name}: {value:.4f}')

def evaluate_performance(labels, probs, threshold=0.5, printout=True, logger=None):
    labels = np.array(labels)
    probs = np.array(probs)

    if logger:
        logger.info(f"Evaluating performance: {len(labels)} samples")

    valid = labels != -1
    labels, probs = labels[valid], probs[valid]
    preds = (probs >= threshold).astype(int)

    if len(np.unique(labels)) < 2:
        if logger:
            logger.error("Not enough label classes for evaluation.")
        return None

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    recall = safe_division(tp, tp + fn)
    specificity = safe_division(tn, tn + fp)
    precision = safe_division(tp, tp + fp)
    f1 = safe_division(2 * precision * recall, precision + recall)

    metrics = [
        accuracy_score(labels, preds),
        roc_auc_score(labels, probs),
        average_precision_score(labels, probs),
        matthews_corrcoef(labels, preds),
        recall, specificity, precision, f1
    ]

    if printout:
        print_metrics(metrics)

    if len(np.unique(labels)) == 2:
        fpr, tpr, _ = roc_curve(labels, probs)
        plot_curve(fpr, tpr, 'False Positive Rate', 'True Positive Rate', 'ROC Curve', f'AUC = {metrics[1]:.3f}')
        precision, recall, _ = precision_recall_curve(labels, probs)
        plot_curve(recall, precision, 'Recall', 'Precision', 'PR Curve', f'AUC = {metrics[2]:.3f}')

    return metrics

def printPerformance(labels, probs, threshold=0.5, printout=True):
    labels = np.array(labels)
    probs = np.array(probs)
    valid = labels != -1
    labels, probs = labels[valid], probs[valid]
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    recall = safe_division(tp, tp + fn)
    specificity = safe_division(tn, tn + fp)
    precision = safe_division(tp, tp + fp)
    f1 = safe_division(2 * precision * recall, precision + recall)

    metrics = [
        accuracy_score(labels, preds),
        roc_auc_score(labels, probs),
        average_precision_score(labels, probs),
        matthews_corrcoef(labels, preds),
        recall, specificity, precision, f1
    ]

    if printout:
        print_metrics(metrics)

    fpr, tpr, _ = roc_curve(labels, probs)
    plot_curve(fpr, tpr, 'False Positive Rate', 'True Positive Rate', 'ROC Curve', f'AUC = {metrics[1]:.3f}')
    precision, recall, _ = precision_recall_curve(labels, probs)
    plot_curve(recall, precision, 'Recall', 'Precision', 'PR Curve', f'AUC = {metrics[2]:.3f}')

    return metrics
