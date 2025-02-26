from __future__ import print_function
import json, time, os, sys, glob
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils.checkpoint
import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools
import csv
from dateutil import parser

def featurize(batch: list, device: torch.device) -> tuple:
    """
    Description:
        Featurizes a batch of protein sequences for model input.
    Args:
        batch (list): A list of dictionaries, each containing sequence and chain information.
        device (torch.device): The device (CPU or GPU) to which the tensors will be moved.
    Returns:
        tuple: A tuple containing:
            - X (torch.Tensor): The feature tensor of shape [B, L_max, 3, 3].
            - S (torch.Tensor): The sequence tensor of shape [B, L_max].
            - mask (torch.Tensor): The mask tensor of shape [B, L].
            - lengths (np.ndarray): The lengths of each sequence in the batch.
            - chain_M (torch.Tensor): The chain mask tensor of shape [B, L_max].
            - residue_idx (torch.Tensor): The residue index tensor of shape [B, L_max].
            - mask_self (torch.Tensor): The self-interaction mask tensor of shape [B, L_max, L_max].
            - chain_encoding_all (torch.Tensor): The chain encoding tensor of shape [B, L_max].
    """
    # Define constants and initialize variables
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)
    L_max = max(lengths)
    
    # Initialize tensors
    X = np.zeros([B, L_max, 3, 3])
    residue_idx = -100 * np.ones([B, L_max], dtype=np.int32)
    chain_M = np.zeros([B, L_max], dtype=np.int32)
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32)
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32)
    S = np.zeros([B, L_max], dtype=np.int32)
    
    # Define chain letters
    init_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    extra_alphabet = [str(i) for i in range(300)]
    chain_letters = init_alphabet + extra_alphabet
    
    # Process each batch item
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        
        # Separate visible and masked chains
        visible_temp_dict = {letter: b[f'seq_chain_{letter}'] for letter in visible_chains}
        masked_temp_dict = {letter: b[f'seq_chain_{letter}'] for letter in masked_chains}
        
        # Update chain lists based on sequence matches
        for masked_letter, masked_seq in masked_temp_dict.items():
            for visible_letter, visible_seq in visible_temp_dict.items():
                if masked_seq == visible_seq:
                    if visible_letter not in masked_chains:
                        masked_chains.append(visible_letter)
                    if visible_letter in visible_chains:
                        visible_chains.remove(visible_letter)
        
        # Shuffle chain order
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)
        
        # Initialize lists for chain data
        x_chain_list, chain_mask_list, chain_seq_list, chain_encoding_list = [], [], [], []
        c, l0, l1 = 1, 0, 0
        
        # Process each chain
        for letter in all_chains:
            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = b[f'coords_chain_{letter}']
            chain_mask = np.zeros(chain_length) if letter in visible_chains else np.ones(chain_length)
            
            # Stack coordinates
            x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}']], axis=1)
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(chain_length))
            
            # Update indices and masks
            l1 += chain_length
            mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1
        
        # Concatenate and pad sequences
        x = np.concatenate(x_chain_list, axis=0)
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, axis=0)
        chain_encoding = np.concatenate(chain_encoding_list, axis=0)
        
        l = len(all_sequence)
        X[i, :l, :, :] = np.pad(x, [[0, L_max - l], [0, 0], [0, 0]], 'constant', constant_values=np.nan)
        chain_M[i, :] = np.pad(m, [0, L_max - l], 'constant', constant_values=0.0)
        chain_encoding_all[i, :] = np.pad(chain_encoding, [0, L_max - l], 'constant', constant_values=0.0)
        
        # Convert sequence to indices
        indices = np.array([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices
    
    # Handle NaNs and convert to tensors
    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, axis=(2, 3))).astype(np.float32)
    X[isnan] = 0.0
    
    # Convert numpy arrays to torch tensors
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long, device=device)
    S = torch.from_numpy(S).to(dtype=torch.long, device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

def loss_nll(S, log_probs, mask):
    """ 
    Description:
        Calculate the negative log likelihood loss.
    Args:
        S (torch.Tensor): Ground truth labels of shape [B, L].
        log_probs (torch.Tensor): Log probabilities of predicted classes of shape [B, L, C].
        mask (torch.Tensor): Mask tensor of shape [B, L] indicating valid positions.
    Returns:
        tuple: A tuple containing:
            - loss (torch.Tensor): The calculated loss for each sequence of shape [B, L].
            - loss_av (torch.Tensor): The average loss across valid positions.
            - true_false (torch.Tensor): A tensor indicating whether the predicted class matches the ground truth.
    """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs, -1)
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false

def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ 
    Description:
        Calculate the smoothed negative log likelihood loss.
    Args:
        S (torch.Tensor): Ground truth labels of shape [B, L].
        log_probs (torch.Tensor): Log probabilities of predicted classes of shape [B, L, C].
        mask (torch.Tensor): Mask tensor of shape [B, L] indicating valid positions.
        weight (float, optional): Smoothing weight for label smoothing. Default is 0.1.
    Returns:
        tuple: A tuple containing:
            - loss (torch.Tensor): The calculated loss for each sequence of shape [B, L].
            - loss_av (torch.Tensor): The average loss across valid positions.
    """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0
    return loss, loss_av

def worker_init_fn(worker_id):
    """Initialize worker with random seed."""
    np.random.seed()

def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    """Process PDB data from data loader."""
    init_alphabet = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    extra_alphabet = [str(i) for i in range(300)]
    chain_alphabet = init_alphabet + extra_alphabet
    
    pdb_dict_list = []
    t0 = time.time()
    
    for _ in range(repeat):
        for step, t in enumerate(data_loader):
            t = {k:v[0] for k,v in t.items()}
            
            if 'label' in t:
                my_dict = {}
                concat_seq = ''
                mask_list = []
                visible_list = []
                
                if len(np.unique(t['idx'])) < 352:
                    for idx in np.unique(t['idx']):
                        letter = chain_alphabet[idx]
                        res = np.argwhere(t['idx']==idx)
                        initial_sequence = "".join(list(np.array(list(t['seq']))[res][0,]))
                        
                        # Handle His-tag sequences
                        if initial_sequence[-6:] == "HHHHHH": res = res[:,:-6]
                        if initial_sequence[0:6] == "HHHHHH": res = res[:,6:]
                        if initial_sequence[-7:-1] == "HHHHHH": res = res[:,:-7]
                        if initial_sequence[-8:-2] == "HHHHHH": res = res[:,:-8]
                        if initial_sequence[-9:-3] == "HHHHHH": res = res[:,:-9]
                        if initial_sequence[-10:-4] == "HHHHHH": res = res[:,:-10]
                        if initial_sequence[1:7] == "HHHHHH": res = res[:,7:]
                        if initial_sequence[2:8] == "HHHHHH": res = res[:,8:]
                        if initial_sequence[3:9] == "HHHHHH": res = res[:,9:]
                        if initial_sequence[4:10] == "HHHHHH": res = res[:,10:]
                        
                        if res.shape[1] >= 4:
                            my_dict['seq_chain_'+letter] = "".join(list(np.array(list(t['seq']))[res][0,]))
                            concat_seq += my_dict['seq_chain_'+letter]
                            
                            if idx in t['masked']:
                                mask_list.append(letter)
                            else:
                                visible_list.append(letter)
                                
                            coords_dict_chain = {}
                            all_atoms = np.array(t['xyz'][res,])[0,]
                            coords_dict_chain['N_chain_'+letter] = all_atoms[:,0,:].tolist()
                            coords_dict_chain['CA_chain_'+letter] = all_atoms[:,1,:].tolist()
                            coords_dict_chain['C_chain_'+letter] = all_atoms[:,2,:].tolist()
                            coords_dict_chain['O_chain_'+letter] = all_atoms[:,3,:].tolist()
                            my_dict['coords_chain_'+letter] = coords_dict_chain
                
                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                        
                    if len(pdb_dict_list) >= num_units:
                        break
    
    return pdb_dict_list

def build_training_clusters(params, debug):
    """Build training, validation, and test clusters from data."""
    val_ids = set([int(l) for l in open(params['VAL']).readlines()])
    test_ids = set([int(l) for l in open(params['TEST']).readlines()])
   
    if debug:
        val_ids = []
        test_ids = []
 
    # Read & clean list.csv
    with open(params['LIST'], 'r') as f:
        reader = csv.reader(f)
        next(reader)
        rows = [[r[0],r[3],int(r[4])] for r in reader
                if float(r[2])<=params['RESCUT'] and
                parser.parse(r[1])<=parser.parse(params['DATCUT'])]
    
    # Compile training and validation sets
    train, valid, test = {}, {}, {}

    if debug:
        rows = rows[:20]
        
    for r in rows:
        if r[2] in val_ids:
            if r[2] in valid:
                valid[r[2]].append(r[:2])
            else:
                valid[r[2]] = [r[:2]]
        elif r[2] in test_ids:
            if r[2] in test:
                test[r[2]].append(r[:2])
            else:
                test[r[2]] = [r[:2]]
        else:
            if r[2] in train:
                train[r[2]].append(r[:2])
            else:
                train[r[2]] = [r[:2]]
                
    if debug:
        valid = train
        
    return train, valid, test

def loader_pdb(item, params):
    """Load PDB data for a given item."""
    if isinstance(item, list):
        pdbid, chid = item[0].split('_')
    else:
        pdbid, chid = item.split('_')
        
    # Get the subdirectory based on the first few characters of the PDB ID
    subdir = pdbid[1:3].lower()
    PREFIX = f"{params['DIR']}/pdb/{subdir}/{pdbid}"
    
    # Load metadata
    if not os.path.isfile(PREFIX+".pt"):
        # Return a dummy tensor for missing files
        return {
            'seq': np.zeros(5, dtype=np.float32),
            'xyz': torch.zeros(5, 3, 3, dtype=torch.float32),
            'idx': torch.zeros(5, dtype=torch.int32),
            'masked': torch.zeros(1, dtype=torch.int32),
            'label': item[0] if isinstance(item, list) else item
        }
        
    meta = torch.load(PREFIX+".pt")
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # Find candidate assemblies containing chid chain
    asmb_candidates = set([a for a,b in zip(asmb_ids,asmb_chains)
                           if chid in b.split(',')])

    # If chain is missing from all assemblies, return chain alone
    if len(asmb_candidates) < 1:
        chain = torch.load(f"{PREFIX}_{chid}.pt")
        L = len(chain['seq'])
        return {
            'seq': chain['seq'].astype(np.float32),
            'xyz': chain['xyz'].to(torch.float32),
            'idx': torch.zeros(L, dtype=torch.int32),
            'masked': torch.zeros(1, dtype=torch.int32),
            'label': item[0] if isinstance(item, list) else item
        }

    # Randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)
    idx = np.where(np.array(asmb_ids)==asmb_i)[0]

    # Load relevant chains
    chains = {c:torch.load(f"{PREFIX}_{c}.pt")
              for i in idx for c in asmb_chains[i].split(',')
              if c in meta['chains']}

    # Generate assembly
    asmb = {}
    for k in idx:
        # Pick k-th xform
        xform = meta[f'asmb_xform{k}']
        u = xform[:,:3,:3]
        r = xform[:,:3,3]

        # Select chains for k-th xform
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        # Transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]['xyz'].to(torch.float32)
                xyz_ru = torch.einsum('bij,raj->brai', u, xyz) + r[:,None,None,:]
                asmb.update({(c,k,i):xyz_i for i,xyz_i in enumerate(xyz_ru)})
            except KeyError:
                return {
                    'seq': np.zeros(5, dtype=np.float32),
                    'xyz': torch.zeros(5, 3, 3, dtype=torch.float32),
                    'idx': torch.zeros(5, dtype=torch.int32),
                    'masked': torch.zeros(1, dtype=torch.int32),
                    'label': item[0] if isinstance(item, list) else item
                }

    # Select chains sharing considerably similarity to chid
    seqid = meta['tm'][chids==chid][0,:,1]
    homo = set([ch_j for seqid_j,ch_j in zip(seqid,chids)
                if seqid_j>params['HOMO']])
                
    # Stack all chains in assembly
    seq, xyz, idx, masked = "", [], [], []
    for counter,(k,v) in enumerate(asmb.items()):
        seq += chains[k[0]]['seq']
        xyz.append(v)
        idx.append(torch.full((v.shape[0],), counter, dtype=torch.int32))
        if k[0] in homo:
            masked.append(counter)

    return {
        'seq': np.array([ord(s) - ord('A') for s in seq], dtype=np.float32),
        'xyz': torch.cat(xyz,dim=0).to(torch.float32),
        'idx': torch.cat(idx,dim=0),
        'masked': torch.tensor(masked, dtype=torch.int32),
        'label': item[0] if isinstance(item, list) else item
    } 

class NoamOpt:
    """
    Description:
        Optimizer wrapper that implements learning rate scheduling.
    Args:
        model_size (int): Size of the model.
        factor (float): Scaling factor for learning rate.
        warmup (int): Warmup period for learning rate.
        optimizer (torch.optim.Optimizer): The optimizer to be wrapped.
        step (int): Current step count.
    """
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer    # The optimizer to be wrapped
        self._step = step            # Current step count
        self.warmup = warmup         # Warmup period for learning rate
        self.factor = factor         # Scaling factor for learning rate
        self.model_size = model_size # Size of the model
        self._rate = 0               # Current learning rate

    @property
    def param_groups(self):
        """Return the parameter groups of the optimizer."""
        return self.optimizer.param_groups

    def step(self):
        """Update parameters and learning rate."""
        self._step += 1                         # Increment the step count
        rate = self.rate()                      # Calculate the new learning rate
        for p in self.optimizer.param_groups:
            p['lr'] = rate                      # Update the learning rate for each parameter group
        self._rate = rate                       # Store the current learning rate
        self.optimizer.step()                   # Perform the optimization step

    def rate(self, step=None):
        """Calculate the learning rate based on the current step."""
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        """Reset the gradients of the optimizer."""
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    """Create a standard optimizer with Noam scheduling."""
    return NoamOpt(
        model_size=d_model,
        factor=2,
        warmup=4000,
        optimizer=torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9),
        step=step
    )

class StructureDataset():
    """
    Description:
        This class is designed to handle protein structure data. It processes a list of protein sequences,
        filtering them based on specific criteria such as maximum sequence length and valid characters.
    """
    def __init__(self, pdb_dict_list, 
                 verbose=True, 
                 truncate=None, 
                 max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            # Check for invalid characters in the sequence
            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                # Check if the sequence length is within the allowed maximum
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Stop processing if the truncate limit is reached
            if truncate is not None and len(self.data) == truncate:
                return

            # Print progress every 1000 entries if verbose is True
            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class StructureLoader():
    """
    Description:
        Custom data loader for protein structures that clusters sequences of similar lengths together
        for efficient batching.
    """
    def __init__(self, dataset, batch_size=100, shuffle=True,
                 collate_fn=lambda x:x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch 