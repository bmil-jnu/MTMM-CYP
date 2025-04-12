import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from rdkit.Chem import Descriptors
from torch_geometric.utils import from_networkx
import networkx as nx
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
from random import Random
import os
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Character to index mapping for SMILES sequences
smi_to_seq = "(.02468@BDFHLNPRTVZ/bdfhlnprt#*%)+-/13579=ACEGIKMOSUWY[]acegimosuy\\"
seq_dict_smi = {v: (i + 1) for i, v in enumerate(smi_to_seq)}
max_seq_smi_len = 100

def one_of_k_encoding_unk(x, allowable_set):
    default_value = allowable_set[-1]
    if x not in allowable_set:
        print(f"Warning: Input {x} not in allowable set. Using default value: {default_value}")
        x = default_value
    return [x == s for s in allowable_set]

def seq_smi(smile, max_seq_smi_len=100):
    indices = np.array([seq_dict_smi.get(ch, 0) for ch in smile[:max_seq_smi_len]], dtype=int)
    return np.pad(indices, (0, max_seq_smi_len - len(indices)), 'constant', constant_values=0)

def atom_features(atom, explicit_H=False, use_chirality=False):
    features = (
        one_of_k_encoding_unk(atom.GetSymbol(), [...]) +
        one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5]) +
        one_of_k_encoding_unk(atom.GetHybridization(), [...]) +
        [atom.GetIsAromatic()])
    return np.array(features, dtype=float)

def filter_valid_smiles(smiles_list):
    """Filter valid SMILES from a list."""
    valid_smiles = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_smiles.append(smi)
        except Exception as e:
            print(f"Invalid SMILES: {smi} | Error: {e}")
    return valid_smiles

def mol2graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        print(f"Warning: Molecule for SMILES {smile} could not be parsed.")
        return None

    features = np.array([atom_features(atom) for atom in mol.GetAtoms()], dtype=float)
    edges = [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]

    g = nx.Graph(edges)
    if g.number_of_edges() == 0:
        print(f"Warning: Graph for SMILES {smile} has no edges.")
        return None

    data = from_networkx(g)
    data.x = torch.tensor(features, dtype=torch.float)
    data.smiles = smile
    return data

class MolDataset(InMemoryDataset):
    def __init__(self, root, dataset, task_type, tasks, logger=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.tasks = tasks
        self.dataset = dataset
        self.task_type = task_type
        self.logger = logger
        super(MolDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices, self.smiles_list = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def process(self):
        dataset_path = os.path.join(self.root, self.dataset)
        df = pd.read_csv(dataset_path)

        smiles_list = filter_valid_smiles(df['Cano_Smile'].values)
        if self.logger:
            self.logger.info(f'Number of valid SMILES in dataset: {len(smiles_list)}')

        data_list = []
        smiles_attr = []

        for i, smi in enumerate(tqdm(smiles_list, desc="Processing SMILES")):
            graph_data = mol2graph(smi)
            if graph_data is None:
                continue

            try:
                label = np.nan_to_num(df.iloc[i][self.tasks].values, nan=-1).astype(np.float32)
            except ValueError as e:
                print(f"Error converting label for SMILES {smi}: {e}")
                continue

            graph_data.y = torch.FloatTensor(label)
            smi_seq = seq_smi(smi, max_seq_smi_len=100)
            graph_data.smil2vec = torch.LongTensor(smi_seq)
            graph_data.smil3D = torch.LongTensor(smi_seq).unsqueeze(0)
            graph_data.atom_masks = torch.ones(graph_data.x.size(0), dtype=torch.float32)

            data_list.append(graph_data)
            smiles_attr.append(smi)

        data, slices = self.collate(data_list)
        torch.save((data, slices, smiles_attr), self.processed_paths[0])

from torch_geometric.data import Batch

class CustomBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        batch = Batch.from_data_list(data_list)
        atom_masks = [data.atom_masks for data in data_list if hasattr(data, 'atom_masks')]
        batch.atom_masks = torch.cat(atom_masks, dim=0) if atom_masks else None
        return batch

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from collections import Counter

def build_multilabel_stratified_loader(data_path, dataset_name, task_type, batch_size, tasks, logger, n_splits=5):
    """
    Construct data loaders using Multilabel Stratified K-Fold split.
    Print number of samples per task in each fold.
    """
    full_dataset = MolDataset(root=data_path, dataset=dataset_name, task_type=task_type, tasks=tasks, logger=logger)
    all_labels = [data.y for data in full_dataset]
    all_labels = torch.stack(all_labels).numpy()

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    fold_data = list(mskf.split(range(len(all_labels)), all_labels))

    train_loaders, val_loaders = [], []
    for fold_idx, (train_index, val_index) in enumerate(fold_data):
        train_data_list = [full_dataset[i] for i in train_index]
        val_data_list = [full_dataset[i] for i in val_index]

        train_loader = GeometricDataLoader(train_data_list, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = GeometricDataLoader(val_data_list, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

        train_counts = Counter()
        val_counts = Counter()
        for data in train_data_list:
            for i, task in enumerate(tasks):
                if data.y[i] != -1:
                    train_counts[task] += 1
        for data in val_data_list:
            for i, task in enumerate(tasks):
                if data.y[i] != -1:
                    val_counts[task] += 1

        if logger:
            logger.info(f"Fold {fold_idx + 1}: Train size: {len(train_data_list)}, Validation size: {len(val_data_list)}")
            logger.info(f"Train data counts: {dict(train_counts)}")
            logger.info(f"Validation data counts: {dict(val_counts)}")
        else:
            print(f"Fold {fold_idx + 1}: Train size: {len(train_data_list)}, Validation size: {len(val_data_list)}")
            print(f"Train data counts: {dict(train_counts)}")
            print(f"Validation data counts: {dict(val_counts)}")

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    return train_loaders, val_loaders

def build_loader(data_path, dataset_names, task_type, batch_size, tasks, logger):
    """
    Build standard train/val/test loaders from provided dataset names.
    """
    train_dataset = MolDataset(root=data_path, dataset=dataset_names['train'], task_type=task_type, tasks=tasks, logger=logger)
    val_dataset = MolDataset(root=data_path, dataset=dataset_names['val'], task_type=task_type, tasks=tasks, logger=logger)
    test_dataset = MolDataset(root=data_path, dataset=dataset_names['test'], task_type=task_type, tasks=tasks, logger=logger)

    train_loader = GeometricDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    test_loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)

    return train_loader, val_loader, test_loader
