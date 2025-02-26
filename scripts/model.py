from __future__ import print_function
import json, time, os, sys, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')

# Import the original model components
def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=4, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        
        self.attention = nn.MultiheadAttention(num_hidden, num_heads, dropout=dropout, batch_first=True)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        # Attention
        dh = self.attention(h_V, h_V, h_V, key_padding_mask=mask_V)[0]
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        return h_V, h_E

class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=4, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])
        
        self.attention = nn.MultiheadAttention(num_hidden, num_heads, dropout=dropout, batch_first=True)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        # Attention
        dh = self.attention(h_V, h_V, h_V, key_padding_mask=mask_V)[0]
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Feed-forward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        return h_V

class ProteinMPNNModel(pl.LightningModule):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
                 hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                 vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1,
                 learning_rate=1e-4, warmup_steps=10000, max_epochs=100):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Model parameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.augment_eps = augment_eps
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs

        # Encoder layers
        self.W_e = nn.Linear(1, edge_features, bias=True)
        self.W_v = nn.Linear(9, hidden_dim, bias=True)
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, edge_features, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, edge_features, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        self.dropout = nn.Dropout(dropout)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def forward(self, seq, xyz, residue_idx, chain_mask, padding_mask=None):
        # Move all inputs to the model's device
        device = self.device
        seq = seq.to(device, non_blocking=True)
        xyz = xyz.to(device, non_blocking=True)
        residue_idx = residue_idx.to(device, non_blocking=True)
        chain_mask = chain_mask.to(device, non_blocking=True)
        if padding_mask is not None:
            padding_mask = padding_mask.to(device, non_blocking=True)
        
        # Convert sequence to long tensor for indexing
        seq = seq.long()
        
        # Get batch size and sequence length
        batch_size, L = xyz.shape[0], xyz.shape[1]
        
        # Reshape xyz from [batch_size, L, 3, 3] to [batch_size * L, 9]
        xyz = xyz.reshape(batch_size * L, -1)  # Flatten the last two dimensions (3x3=9)
        
        # Initial embeddings
        h_V = self.W_v(xyz)  # Node features from coordinates
        h_V = h_V.reshape(batch_size, L, -1)  # Reshape back to [batch_size, L, hidden_dim]
        
        # Edge features from residue indices
        # First create pairwise distances between residue indices
        idx_dists = (residue_idx.unsqueeze(-1) - residue_idx.unsqueeze(-2)).float()  # [batch_size, L, L]
        h_E = self.W_e(idx_dists.unsqueeze(-1))  # [batch_size, L, L, edge_features]
        
        # Create attention mask from residue indices and padding
        mask_attend = (residue_idx.unsqueeze(-1) == residue_idx.unsqueeze(-2)).float()
        if padding_mask is not None:
            # Add padding mask to attention mask
            padding_attend = padding_mask.unsqueeze(-1) & padding_mask.unsqueeze(-2)  # [batch_size, L, L]
            mask_attend = mask_attend * padding_attend.float()
        
        # Encoder
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, residue_idx, mask_V=~padding_mask if padding_mask is not None else None, mask_attend=mask_attend)
            
        # Decoder
        for layer in self.decoder_layers:
            h_V = layer(h_V, h_E, mask_V=~padding_mask if padding_mask is not None else None, mask_attend=mask_attend)
            
        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs

    def training_step(self, batch, batch_idx):
        seq, xyz, residue_idx, chain_mask, labels, padding_mask = batch
        log_probs = self(seq, xyz, residue_idx, chain_mask, padding_mask)
        
        # Apply padding mask to loss
        loss = self.criterion(log_probs.view(-1, log_probs.size(-1)), seq.view(-1))
        loss = loss.view(seq.shape)
        if padding_mask is not None:
            loss = loss * padding_mask
        loss = loss.sum() / (padding_mask.sum() if padding_mask is not None else loss.numel())
        
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, xyz, residue_idx, chain_mask, labels, padding_mask = batch
        log_probs = self(seq, xyz, residue_idx, chain_mask, padding_mask)
        
        # Apply padding mask to loss
        loss = self.criterion(log_probs.view(-1, log_probs.size(-1)), seq.view(-1))
        loss = loss.view(seq.shape)
        if padding_mask is not None:
            loss = loss * padding_mask
        loss = loss.sum() / (padding_mask.sum() if padding_mask is not None else loss.numel())
        
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        } 