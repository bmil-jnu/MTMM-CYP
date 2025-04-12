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

def train(epoch, model, criterion, train_loader, optimizer, lr_scheduler, device, task_type='classification', metric='auc', logger=None):
    model.train()
    losses = []
    task_losses = defaultdict(float)
    y_pred_list = {i: [] for i in range(5)}
    y_label_list = {i: [] for i in range(5)}
    num_tasks = 5

    for batch_idx, batch in enumerate(train_loader):
        data = batch.to(device)

        # Forward pass
        _, task_outputs = model({
            'fp': data.smil3D,
            'graph': data,
            'conv': data.smil2vec if hasattr(data, 'smil2vec') else torch.zeros((data.num_graphs, 100)).to(device)
        })

        optimizer.zero_grad()
        loss = 0.0

        # Reshape labels to [batch_size, num_tasks]
        y_labels = data.y
        batch_size = y_labels.size(0) // num_tasks
        y_labels = y_labels.view(batch_size, num_tasks)

        # Compute loss for each task
        for i in range(num_tasks):
            y_pred = task_outputs[i][:batch_size].squeeze(-1)
            y_label = y_labels[:batch_size, i]
            valid_idx = y_label != -1
            if valid_idx.sum().item() == 0:
                continue

            y_pred = y_pred[valid_idx]
            y_label = y_label[valid_idx]

            if y_pred.size(0) == 0 or y_label.size(0) == 0:
                continue

            y_label = y_label.view(-1)
            task_loss = criterion(y_pred, y_label.float())
            loss += task_loss
            task_losses[f'task_{i}'] += task_loss.item() * valid_idx.sum().item()

            y_pred_list[i].extend(torch.sigmoid(y_pred).detach().cpu().numpy())
            y_label_list[i].extend(y_label.cpu().numpy())

        # Backward pass and optimizer step
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        losses.append(loss.item())

    # Normalize task losses
    dataset_size = len(train_loader.dataset)
    for task in task_losses:
        task_losses[task] /= dataset_size

    # Evaluate performance
    train_results = []
    metric_func = get_metric_func(metric=metric)
    for i in range(num_tasks):
        if len(y_label_list[i]) == 0:
            continue
        train_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_results = np.nanmean(train_results)
    trn_loss = np.mean(losses)
    print(f'Epoch: {epoch}, Train Loss: {trn_loss:.4f}')
    return trn_loss, avg_results

@torch.no_grad()
def validate(model, criterion, val_loader, device, task_type='classification', metric='auc', logger=None, eval_mode=False, epoch=None, seed=None):
    model.eval()
    start = time.time()
    losses = []
    task_losses = defaultdict(float)
    y_pred_list = {i: [] for i in range(5)}
    y_label_list = {i: [] for i in range(5)}
    num_tasks = 5

    for batch_idx, batch in enumerate(val_loader):
        data = batch.to(device)

        # Forward pass
        _, task_outputs = model({
            'fp': data.smil3D,
            'graph': data,
            'conv': data.smil2vec if hasattr(data, 'smil2vec') else torch.zeros((data.num_graphs, 100)).to(device)
        })

        # Reshape labels
        y_labels = data.y
        batch_size = y_labels.size(0) // num_tasks
        y_labels = y_labels.view(batch_size, num_tasks)

        # Compute task losses
        for i in range(num_tasks):
            y_pred = task_outputs[i][:batch_size].squeeze(-1)
            y_label = y_labels[:batch_size, i]
            valid_idx = y_label != -1
            if valid_idx.sum().item() == 0:
                continue

            y_pred = y_pred[valid_idx]
            y_label = y_label[valid_idx]
            if y_pred.size(0) == 0 or y_label.size(0) == 0:
                continue

            y_label = y_label.view(-1)
            task_loss = criterion(y_pred, y_label.float())
            task_losses[f'task_{i}'] += task_loss.item() * valid_idx.sum().item()
            losses.append(task_loss.item())

            y_pred_list[i].extend(torch.sigmoid(y_pred).detach().cpu().numpy())
            y_label_list[i].extend(y_label.cpu().numpy())

    dataset_size = len(val_loader.dataset)
    for task in task_losses:
        task_losses[task] /= dataset_size

    # Evaluate performance
    val_results = []
    metric_func = get_metric_func(metric=metric)
    for i in range(num_tasks):
        if len(y_label_list[i]) == 0:
            continue
        val_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_val_results = np.nanmean(val_results) if val_results else float('nan')
    val_loss = np.mean(losses)
    duration = time.time() - start
    print(f'Epoch: {epoch}, Validation Loss: {val_loss:.4f}, Duration: {duration:.2f}s')
    return val_loss, avg_val_results

def test(model, criterion, test_loader, device, task_type='classification', metric='auc', logger=None, drop_last=True):
    model.eval()
    start = time.time()
    losses = []
    task_losses = defaultdict(float)
    y_pred_list = {i: [] for i in range(5)}
    y_label_list = {i: [] for i in range(5)}
    num_tasks = 5

    for batch_idx, batch in enumerate(test_loader):
        data = batch.to(device)

        # Forward pass
        _, task_outputs = model({
            'fp': data.smil3D,
            'graph': data,
            'conv': data.smil2vec if hasattr(data, 'smil2vec') else torch.zeros((data.num_graphs, 100)).to(device)
        })

        # Reshape labels
        y_labels = data.y
        batch_size = y_labels.size(0) // num_tasks
        y_labels = y_labels.view(batch_size, num_tasks)

        # Compute task losses
        for i in range(num_tasks):
            y_pred = task_outputs[i][:batch_size].squeeze(-1)
            y_label = y_labels[:batch_size, i]
            valid_idx = y_label != -1
            if valid_idx.sum().item() == 0:
                continue

            y_pred = y_pred[valid_idx]
            y_label = y_label[valid_idx]
            if y_pred.size(0) == 0 or y_label.size(0) == 0:
                continue

            y_label = y_label.view(-1)
            task_loss = criterion(y_pred, y_label.float())
            task_losses[f'task_{i}'] += task_loss.item() * valid_idx.sum().item()
            losses.append(task_loss.item())

            y_pred_list[i].extend(torch.sigmoid(y_pred).detach().cpu().numpy())
            y_label_list[i].extend(y_label.cpu().numpy())

    dataset_size = len(test_loader.dataset)
    for task in task_losses:
        task_losses[task] /= dataset_size

    # Evaluate performance
    test_results = []
    metric_func = get_metric_func(metric=metric)
    for i in range(num_tasks):
        if len(y_label_list[i]) == 0:
            continue
        test_results.append(metric_func(y_label_list[i], y_pred_list[i]))

    avg_test_results = np.nanmean(test_results)
    test_loss = np.mean(losses)
    duration = time.time() - start
    print('====> Average Test Loss: {:.4f}'.format(test_loss / len(test_loader.dataset)))
    return test_loss, avg_test_results