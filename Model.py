import os
import logging
import time
import random
import numpy as np
import math
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch.nn.modules.batchnorm import _BatchNorm
from torch_geometric import data
from torch_geometric.data import Batch, Data as GeometricData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, GraphConv, TopKPooling, GATConv
from torch.nn import init
from torch.nn.parameter import Parameter
from torch import Tensor
from MTMM_utile import create_logger

# ---------------------------------
# Sequence Module
# ---------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)

        # Extend positional encoding if sequence length exceeds precomputed max_len
        if seq_len > self.max_len:
            pe = torch.zeros(seq_len, self.d_model).to(x.device)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1).to(x.device)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
        else:
            pe = self.pe[:, :seq_len, :].to(x.device)

        x = x + pe
        return self.dropout(x)

class TrfmSeq2seq(nn.Module):
    def __init__(self, input_dim=100, hidden_size=512, num_head=4, n_layers=4, dropout=0.5, vocab_num=0, device=None, recons=False):
        super(TrfmSeq2seq, self).__init__()
        self.hidden_size = hidden_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.recons = recons

        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        transformer = nn.Transformer(d_model=hidden_size, nhead=num_head, num_encoder_layers=n_layers,
                                     num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder

        self.linear_input = nn.Linear(input_dim, hidden_size)
        self.out = nn.Linear(hidden_size, input_dim)
        self.recon_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    def forward(self, src):
        src = src.float().to(self.device)
        if src.dim() == 2:
            src = src.unsqueeze(1)

        src = self.linear_input(src)
        src = self.pos_encoder(src)
        hidden = self.encoder(src)

        loss = 0
        if self.recons:
            out = self.decoder(src, hidden)
            out = F.log_softmax(self.out(out), dim=-1)
            loss = self.recon_loss(out.view(-1, self.vocab_num), src.view(-1))

        return hidden, loss

# ---------------------------------
# Graph Module
# ---------------------------------
class GraphModule(nn.Module):
    def __init__(self, in_channels=77, out_channels=256, dropout=0.5, device=None):
        super(GraphModule, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = GATConv(in_channels, 256, heads=4).to(self.device)
        self.conv2 = GATConv(256 * 4, 256, heads=4).to(self.device)

        self.bn1 = nn.BatchNorm1d(256 * 4).to(self.device)
        self.bn2 = nn.BatchNorm1d(256 * 4).to(self.device)

        self.fc_final = nn.Linear(256 * 4, out_channels).to(self.device)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x.to(self.device), data.edge_index.to(self.device), data.batch.to(self.device)
        x = self.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.conv2(x, edge_index)))
        x = global_max_pool(x, batch)
        x = self.fc_final(x)
        return x

# ---------------------------------
# Convolution Module
# ---------------------------------
class SE_Block(nn.Module):
    def __init__(self, in_planes):
        super(SE_Block, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.conv1 = nn.Conv1d(in_planes, in_planes // 16, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_planes // 16, in_planes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(self.avgpool(x))))
        return self.sigmoid(x)

class ConvModule(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=256, embed_dim=128, output_dim=256, dropout=0.5, device=None):
        super(ConvModule, self).__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()

        self.embedding_xt_smile = nn.Embedding(100, embed_dim).to(self.device)

        self.conv_xt2 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=2).to(self.device)
        self.fc_xt2 = nn.Linear(n_filters * 127, output_dim).to(self.device)

        self.conv_xt4 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=4).to(self.device)
        self.fc_xt4 = nn.Linear(n_filters * 125, output_dim).to(self.device)

        self.conv_xt8 = nn.Conv1d(in_channels=100, out_channels=n_filters, kernel_size=8).to(self.device)
        self.SE1 = SE_Block(n_filters).to(self.device)
        self.fc_xt8 = nn.Linear(n_filters * 121, output_dim).to(self.device)

        total_input_size = (n_filters * 127) + (n_filters * 125) + (n_filters * 121)
        self.fc3 = nn.Linear(total_input_size, output_dim).to(self.device)
        self.out = nn.Linear(output_dim, self.n_output).to(self.device)

    def forward(self, smil2vec):
        smil2vec = smil2vec.to(self.device)
        batch_size = smil2vec.size(0) // 100
        smil2vec = smil2vec.view(batch_size, -1)
        embedded_xt1 = self.embedding_xt_smile(smil2vec)

        conv_xt2 = self.conv_xt2(embedded_xt1)
        conv_xt2 = self.relu(conv_xt2) * self.SE1(conv_xt2)

        conv_xt4 = self.conv_xt4(embedded_xt1)
        conv_xt4 = self.relu(conv_xt4) * self.SE1(conv_xt4)

        conv_xt8 = self.conv_xt8(embedded_xt1)
        conv_xt8 = self.relu(conv_xt8) * self.SE1(conv_xt8)

        x2 = conv_xt2.view(conv_xt2.size(0), -1)
        x4 = conv_xt4.view(conv_xt4.size(0), -1)
        x8 = conv_xt8.view(conv_xt8.size(0), -1)

        x = torch.cat([x2, x4, x8], dim=1)
        x = self.fc3(x)
        return x

    # ---------------------------------
# Fusion Module for Multi-modal Representations
# ---------------------------------
class WeightFusion(nn.Module):
    def __init__(self, feat_views, feat_dim, bias=True, device=None):
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim

        # Learnable weight for each input modality
        self.weight = Parameter(torch.empty(feat_views, feat_dim))

        # Optional bias for fused output
        self.bias = Parameter(torch.empty(feat_dim)) if bias else None

        self.reset_parameters()
        self.dropout = nn.Dropout(p=0.5)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.feat_views)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input = self.dropout(input)
        weighted_sum = torch.sum(input * self.weight.unsqueeze(1), dim=0)
        if self.bias is not None:
            weighted_sum += self.bias
        return weighted_sum

# ---------------------------------
# MTMM-CYP
# ---------------------------------
class MTMM(nn.Module):
    def __init__(self, filter_num=256, device=None, fusion_type=3):
        super().__init__()
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Fingerprint encoder: transformer-based
        self.fp_encoder = TrfmSeq2seq(input_dim=100, hidden_size=256, num_head=4, n_layers=4,
                                      dropout=0.5, vocab_num=0, device=device, recons=False)
        self.fp_fc = nn.Linear(256, 512)

        # Graph encoder: GAT-based
        self.graph_encoder = GraphModule()
        self.graph_fc = nn.Linear(256, 512)

        # SMILES encoder: CNN with SE block
        self.smi_encoder = ConvModule()
        self.smi_fc = nn.Linear(256, 512)

        # Feature fusion module
        self.fusion_type = fusion_type
        feat_views = 3  # fingerprint, graph, SMILES
        feat_dim = 512
        self.fusion = WeightFusion(feat_views, feat_dim)

        # Task-specific heads for 5 CYP450 isoforms
        self.num_tasks = 5
        pooled_output_size = feat_dim
        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pooled_output_size, 512),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(0.5),
            ) for _ in range(self.num_tasks)
        ])
        self.outputs = nn.ModuleList([nn.Linear(256, 1) for _ in range(self.num_tasks)])

    def forward(self, data):
        # Fingerprint input handling
        fp_x = data['fp'].to(self.device)
        if fp_x.dim() == 2:
            if fp_x.size(-1) < 100:
                padding = torch.zeros(fp_x.size(0), 100 - fp_x.size(-1)).to(fp_x.device)
                fp_x = torch.cat([fp_x, padding], dim=-1)
            elif fp_x.size(-1) > 100:
                fp_x = fp_x[:, :100]
        elif fp_x.dim() == 3:
            if fp_x.size(1) < 100:
                padding = torch.zeros(fp_x.size(0), 100 - fp_x.size(1), fp_x.size(2)).to(fp_x.device)
                fp_x = torch.cat([fp_x, padding], dim=1)
            elif fp_x.size(1) > 100:
                fp_x = fp_x[:, :100, :]

        # Fingerprint encoding
        fp_x, _ = self.fp_encoder(fp_x)
        fp_x = fp_x.mean(dim=1)
        fp_x = self.fp_fc(fp_x)

        # Graph encoding
        graph_x = self.graph_encoder(data['graph'])
        graph_x = self.graph_fc(graph_x)

        # SMILES encoding
        if 'conv' in data:
            smi_x = self.smi_encoder(data['conv'])
            smi_x = self.smi_fc(smi_x)
        else:
            # If missing, use placeholder tensor
            smi_x = torch.zeros(graph_x.size(0), 128).to(self.device)

        # Align batch sizes across modalities
        min_batch_size = min(fp_x.size(0), graph_x.size(0), smi_x.size(0))
        fp_x = fp_x[:min_batch_size]
        graph_x = graph_x[:min_batch_size]
        smi_x = smi_x[:min_batch_size]

        # Fuse features from all modalities
        pooled_outputs = torch.stack([fp_x, graph_x, smi_x], dim=0)
        pooled_output = self.fusion(pooled_outputs)

        # Multi-task prediction
        final_outputs = []
        for head, output_layer in zip(self.task_heads, self.outputs):
            h = head(pooled_output)
            task_output = output_layer(h).view(-1, 1)
            final_outputs.append(task_output)

        return pooled_output, tuple(final_outputs)