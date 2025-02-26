import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Dict, List

class ProteinDataset(Dataset):
    def __init__(self, pdb_ids: List[str], pdb_loader, pdb_dict: Dict, params: Dict):
        self.pdb_ids = []
        self.items = []
        for cluster_id, items in pdb_dict.items():
            for item in items:
                self.pdb_ids.append(f"{item[0]}")
                self.items.append(item)
        self.pdb_loader = pdb_loader
        self.pdb_dict = pdb_dict
        self.params = params

    def __len__(self):
        return len(self.pdb_ids)

    def __getitem__(self, idx):
        item = self.items[idx]
        data = self.pdb_loader(item, self.params)
        if isinstance(data, dict):
            # Convert data to list format for featurize
            batch = [{
                'seq': data['seq'],
                'coords_chain_A': {
                    'N_chain_A': data['xyz'][:, 0, :].tolist(),
                    'CA_chain_A': data['xyz'][:, 1, :].tolist(),
                    'C_chain_A': data['xyz'][:, 2, :].tolist()
                },
                'seq_chain_A': ''.join([chr(ord('A') + int(s)) for s in data['seq']]),
                'name': data['label'],
                'masked_list': ['A'] if len(data['masked']) > 0 else [],
                'visible_list': [] if len(data['masked']) > 0 else ['A'],
                'num_of_chains': 1
            }]
            # Featurize the data
            from utils import featurize
            X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize(batch, torch.device('cpu'))
            return S[0], X[0], residue_idx[0], chain_M[0], data['label']
        return data

class ProteinDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        num_workers: int = 4,
        rescut: float = 3.5,
        homo: float = 0.70,
        debug: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Data parameters
        self.params = {
            "LIST": f"{data_dir}/list.csv",
            "VAL": f"{data_dir}/valid_clusters.txt",
            "TEST": f"{data_dir}/test_clusters.txt",
            "DIR": data_dir,
            "DATCUT": "2021-Aug-02",
            "RESCUT": rescut,
            "HOMO": homo
        }
        
        self.debug = debug
        self.train_clusters = None
        self.valid_clusters = None
        self.test_clusters = None

    def setup(self, stage: Optional[str] = None):
        from utils import build_training_clusters, loader_pdb
        
        # Build clusters only once
        if self.train_clusters is None:
            self.train_clusters, self.valid_clusters, self.test_clusters = build_training_clusters(
                self.params,
                self.debug
            )
        
        if stage == 'fit' or stage is None:
            self.train_dataset = ProteinDataset(
                list(self.train_clusters.keys()),
                loader_pdb,
                self.train_clusters,
                self.params
            )
            self.val_dataset = ProteinDataset(
                list(self.valid_clusters.keys()),
                loader_pdb,
                self.valid_clusters,
                self.params
            )
            
        if stage == 'test':
            self.test_dataset = ProteinDataset(
                list(self.test_clusters.keys()),
                loader_pdb,
                self.test_clusters,
                self.params
            )

    def collate_fn(self, batch):
        # Find max length in the batch
        max_len = max(x[0].size(0) for x in batch)
        
        # Initialize padded tensors (explicitly on CPU)
        B = len(batch)
        S = torch.zeros(B, max_len, dtype=torch.long)
        X = torch.zeros(B, max_len, 3, 3, dtype=torch.float)
        residue_idx = torch.zeros(B, max_len, dtype=torch.long)
        chain_M = torch.zeros(B, max_len, dtype=torch.float)
        mask = torch.zeros(B, max_len, dtype=torch.bool)
        labels = []
        
        # Fill padded tensors
        for i, (s, x, r, m, label) in enumerate(batch):
            L = s.size(0)
            # Ensure tensors are on CPU
            S[i, :L] = s.cpu()
            X[i, :L] = x.cpu()
            residue_idx[i, :L] = r.cpu()
            chain_M[i, :L] = m.cpu()
            mask[i, :L] = True
            labels.append(label)
        
        return S, X, residue_idx, chain_M, labels, mask

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing
            pin_memory=False,
            collate_fn=self.collate_fn
        ) 