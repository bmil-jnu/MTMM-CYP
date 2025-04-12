from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import display, SVG
import svgutils.transform as sg 
from rdkit.Chem import Draw

# Normalize importance values to [0, 1] range
def normalize_importance(importance_values):
    return (importance_values - importance_values.min()) / (importance_values.max() - importance_values.min() + 1e-3)

# Generate atom colors based on threshold
def get_atom_colors(importance_values, threshold=0.3):
    colors = {}
    for i, val in enumerate(importance_values):
        if val >= threshold:
            colors[i] = (1.0, 0.0, 0.0, 0.8)  # Red in RGBA
    return colors

# Visualize Grad-CAM heatmap overlay on molecule structure
def visualize_fusion_gradcam(grad_cam, mol, atom_importance=None, save_path=None):
    num_atoms = mol.GetNumAtoms()
    grad_cam_data = grad_cam.detach().cpu().numpy()[:num_atoms]

    if atom_importance is not None:
        fused_importance = grad_cam_data + atom_importance[:num_atoms].detach().cpu().numpy()
    else:
        fused_importance = grad_cam_data

    atom_importance = normalize_importance(fused_importance)
    atom_colors = get_atom_colors(atom_importance, threshold=0.3)

    drawer = rdMolDraw2D.MolDraw2DSVG(500, 500)
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=list(atom_colors.keys()), highlightAtomColors=atom_colors)
    drawer.FinishDrawing()

    svg = drawer.GetDrawingText()
    display(SVG(svg))

    if save_path:
        with open(save_path, "w") as f:
            f.write(svg)
        print(f"Saved Grad-CAM visualization to {save_path}")

# GradCAM class to handle hook registration and heatmap generation
class GradCAM:
    def __init__(self, model, target_module, target_output_index=0):
        self.model = model
        self.target_module = target_module
        self.target_output_index = target_output_index
        self.gradients = None
        self.activations = None

        self.target_module.register_forward_hook(self._forward_hook)
        self.target_module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output[self.target_output_index] if isinstance(output, tuple) else output
        print(f"Forward hook set: {self.activations.shape}")

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[self.target_output_index] if isinstance(grad_output, tuple) else grad_output[0]
        print(f"Backward hook set: {self.gradients.shape}")

    def generate_heatmap(self):
        if self.gradients is None or self.activations is None:
            raise ValueError("Gradients or activations not set. Check hook registration.")

        weights = self.gradients.mean(dim=0)
        grad_cam = (weights.unsqueeze(0) * self.activations).sum(dim=1)
        grad_cam = torch.nn.functional.relu(grad_cam)
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() + 1e-8)

        return grad_cam

from sklearn.model_selection import StratifiedKFold
import torch
import numpy as np

# Main training loop with multi-fold and optional Grad-CAM saving
def main_train(output_dir="output", tag="default", seed=1, batch_size=64, task_type='classification', 
               metric='prc', optimizer_type='adam', weight_decay=1e-4, scheduler_type='reduce', factor=0.7, patience=4, min_lr=1e-5, 
               eval_mode=False, base_lr=1e-5, n_splits=10):

    seed_set(seed)
    logger = create_logger(output_dir=output_dir, tag=tag)
    logger.info("Starting training...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    data_path = 'D:/bmil/p450/MTMM/'
    dataset_names = {
        'train': 'train_VAL.csv',
        'val': 'val.csv',
        'test': 'test.csv'
    }
    tasks = ['1a2', '2c9', '2c19', '2d6', '3a4']

    train_loaders, val_loaders = build_multilabel_stratified_loader(
        data_path=data_path, 
        dataset_name=dataset_names['train'], 
        task_type=task_type, 
        batch_size=batch_size, 
        tasks=tasks, 
        logger=logger, 
        n_splits=n_splits
    )

    _, _, test_loader = build_loader(
        data_path=data_path, 
        dataset_names=dataset_names, 
        task_type=task_type, 
        batch_size=batch_size, 
        tasks=tasks, 
        logger=logger
    )

    model = MTMM(device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    early_stopping = EarlyStopping(patience=7, delta=0.001, monitor='loss', path='best_model_loss.pt', verbose=True)
    criterion = FocalLoss()
    best_score = 0 if task_type == 'classification' else float('inf')

    trn_losses = []
    val_losses = []

    for fold_idx, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
        logger.info(f"Starting Fold {fold_idx + 1}/{n_splits}")
        fold_train_loss_list = []
        fold_validation_loss_list = []

        for epoch in range(1, 1000):
            logger.info(f"Starting epoch {epoch}/1000")

            trn_loss, trn_score = train(epoch, model, criterion, train_loader, optimizer, lr_scheduler, device, task_type, metric, logger)
            val_loss, val_score = validate(model, criterion, val_loader, device, task_type, metric, logger)

            fold_train_loss_list.append(trn_loss)
            fold_validation_loss_list.append(val_loss)

            if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                lr_scheduler.step(val_loss)

            if (task_type == 'classification' and val_score > best_score) or (task_type == 'regression' and val_score < best_score):
                best_score = val_score
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1

            early_stopping(val_score if early_stopping.monitor == 'auc' else val_loss, model)

            if early_stopping.early_stop:
                logger.info(f"Early stopping triggered for Fold {fold_idx + 1}")
                break

        trn_losses.append(np.mean(fold_train_loss_list))
        val_losses.append(np.mean(fold_validation_loss_list))

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
                'conv': batch.smil2vec if hasattr(batch, 'smil2vec') else None,
                'atom_masks': batch.atom_masks 
            }
            _, _, _, preds = model(input_data)
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

    return model, trn_losses, val_losses, test_loader

# Run Grad-CAM on test dataset and save visualizations
def test_with_gradcam(model, test_loader, device, save_dir="gradcam_results"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    fusion_grad_cam = GradCAM(model=model, target_module=model.fusion, target_output_index=0)

    for batch_idx, batch in enumerate(test_loader):
        data = batch.to(device)
        input_data = {
            'fp': data.smil3D,
            'graph': data,
            'conv': data.smil2vec if hasattr(data, 'smil2vec') else torch.zeros((data.num_graphs, 100)).to(device),
            'atom_masks': data.atom_masks
        }

        pooled_output, importance_scores, top_substructures, preds = model(input_data)
        target_task_output = preds[0]
        target = torch.ones_like(target_task_output).to(device)
        loss = F.mse_loss(target_task_output, target)
        model.zero_grad()
        loss.backward()

        smiles = data.smiles if hasattr(data, 'smiles') else "Unknown SMILES"
        smiles = smiles[0] if isinstance(smiles, list) else smiles

        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Skipping invalid SMILES: {smiles}")
            continue

        print(f"\nBatch {batch_idx}, SMILES: {smiles}")
        fusion_heatmap = fusion_grad_cam.generate_heatmap()

        save_path = os.path.join(save_dir, f"gradcam_batch{batch_idx}.svg")
        visualize_fusion_gradcam(fusion_heatmap, mol, save_path=save_path)

if __name__ == "__main__":
    model, trn_losses, val_losses, test_loader = main_train()
    print("Starting Grad-CAM visualization...")
    test_with_gradcam(
        model=model,
        test_loader=test_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir="gradcam_results"
    )
