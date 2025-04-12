import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
import torch_geometric
from torch_geometric import data
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from model import MTMM
from evaluate import printPerformance
from utile import evaluate_performance, seed_set, create_logger, build_optimizer, build_scheduler, EarlyStopping, get_metric_func
from Focal_loss import FocalLoss
from dataset import build_loader, build_multilabel_stratified_loader
from sklearn.model_selection import StratifiedKFold


def main_train(output_dir="output", tag="default", seed=1, batch_size=64, task_type='classification',
               metric='prc', optimizer_type='adam', weight_decay=1e-4, scheduler_type='reduce', factor=0.7, patience=4, min_lr=1e-5,
               eval_mode=False, base_lr=1e-5, n_splits=10):

    # Set random seed for reproducibility
    seed_set(seed)

    # Initialize logger
    logger = create_logger(output_dir=output_dir, tag=tag)
    logger.info("Starting training...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Define dataset paths and tasks
    data_path = 'D:/bmil/p450/MTMM/'
    dataset_names = {
        'train': 'train_VAL.csv',
        'val': 'val.csv',
        'test': 'test.csv'
    }
    tasks = ['1a2', '2c9', '2c19', '2d6', '3a4']

    # Create stratified K-fold data loaders for training and validation
    train_loaders, val_loaders = build_multilabel_stratified_loader(
        data_path=data_path,
        dataset_name=dataset_names['train'],
        task_type=task_type,
        batch_size=batch_size,
        tasks=tasks,
        logger=logger,
        n_splits=n_splits
    )

    # Create test loader separately
    _, _, test_loader = build_loader(
        data_path=data_path,
        dataset_names=dataset_names,
        task_type=task_type,
        batch_size=batch_size,
        tasks=tasks,
        logger=logger
    )

    # Initialize model and optimizer
    model = MTMM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    early_stopping = EarlyStopping(patience=8, delta=0.001, monitor='loss', path='best_model_loss.pt', verbose=True)
    early_stop_cnt = 0
    criterion = FocalLoss()

    # Set best_score depending on task type
    best_score = 0 if task_type == 'classification' else float('inf')

    trn_losses = []
    val_losses = []

    # Loop through K folds
    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info(f"Starting Fold {fold_idx + 1}/{n_splits}")

        fold_train_loss_list = []
        fold_validation_loss_list = []

        # Training epochs per fold
        for epoch in range(1, 1000):
            logger.info(f"Starting epoch {epoch}/1000")

            # Training
            trn_loss, trn_score = train(
                epoch=epoch,
                model=model,
                criterion=criterion,
                train_loader=train_loader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger
            )

            # Validation
            val_loss, val_score = validate(
                model=model,
                criterion=criterion,
                val_loader=val_loader,
                device=device,
                task_type=task_type,
                metric=metric,
                logger=logger
            )

            fold_train_loss_list.append(trn_loss)
            fold_validation_loss_list.append(val_loss)

            # Update learning rate scheduler
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                lr_scheduler.step(val_loss)

            # Save best model and apply early stopping
            if (task_type == 'classification' and val_score > best_score) or (task_type == 'regression' and val_score < best_score):
                best_score, best_epoch = val_score, epoch
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            if early_stopping.monitor == 'auc':
                early_stopping(val_score, model)
            else:
                early_stopping(val_loss, model)

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered for Fold {fold_idx + 1}")
                break

        # Record average loss per fold
        trn_losses.append(np.mean(fold_train_loss_list))
        val_losses.append(np.mean(fold_validation_loss_list))

    # Start test evaluation
    logger.info("Starting test loop...")
    test_labels = defaultdict(list)
    test_probs = defaultdict(list)

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            labels = batch.y.view(batch.num_graphs, -1).cpu().numpy()

            input_data = {
                'fp': batch.x,
                'graph': batch,
                'conv': batch.smil2vec if hasattr(batch, 'smil2vec') else None
            }

            _, preds = model(input_data)
            probs = torch.cat([torch.sigmoid(pred).detach().cpu() for pred in preds], dim=1)

            if labels.shape != probs.shape:
                raise ValueError(f"Shape mismatch: labels shape {labels.shape}, probs shape {probs.shape}")

            for i, task_name in enumerate(tasks):
                test_labels[task_name].extend(labels[:, i].tolist())
                test_probs[task_name].extend(probs[:, i].tolist())

        for task in test_labels.keys():
            logger.info(f"Performance for {task}:")
            printPerformance(test_labels[task], test_probs[task])
            logger.info("-" * 50)

        return model, trn_losses, val_losses

if __name__ == "__main__":
    model, trn_losses, val_losses = main_train()
    torch.save(model.state_dict(), 'save_MTMM.pth')
    print("Model saved to save_MTMM.pth")
