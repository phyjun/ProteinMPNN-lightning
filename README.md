# ProteinMPNN-Lightning

This is a PyTorch Lightning implementation of the ProteinMPNN (Message Passing Neural Network) model for protein sequence design.

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [File Structure](#file-structure)
- [Reading PDB Files](#reading-pdb-files)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [License](#license)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ProteinMPNN-lightning.git
cd ProteinMPNN-lightning
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data Preparation

1. Download the sample dataset:
```bash
cd data
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz
tar -xzf pdb_2021aug02_sample.tar.gz
```

2. Dataset structure:
```
data/
├── pdb_2021aug02_sample/
│   ├── list.csv                # List of PDB chains and their properties
│   ├── valid_clusters.txt      # Validation set cluster IDs
│   ├── test_clusters.txt       # Test set cluster IDs
│   └── pdb/                    # PDB files organized by prefix
│       ├── a1/
│       ├── a2/
│       └── ...
```

3. File formats:
   - `list.csv`: Contains metadata for each chain
     ```
     CHAINID,DEPOSITION,RESOLUTION,HASH,CLUSTER,SEQUENCE
     1a3x_A,2020-01-01,2.5,abc123,1,SEQUENCE...
     ```
   - `valid_clusters.txt` and `test_clusters.txt`: One cluster ID per line
   - PDB files are stored as PyTorch tensors (.pt files)

## File Structure

### PDB File Organization

Each PDB entry consists of two types of files:

1. **PDBID_CHAINID.pt** (e.g., 1l3a_A.pt)
   - Contains chain-specific data
   - Located at: `pdb/<prefix>/<pdbid>_<chainid>.pt`
   - Fields:
     ```python
     {
         'seq': str,              # Amino acid sequence
         'xyz': torch.Tensor,     # Shape [L, 14, 3] - Atomic coordinates
         'mask': torch.Tensor,    # Shape [L, 14] - Atom mask
         'bfac': torch.Tensor,    # Shape [L, 14] - B-factors
         'occ': torch.Tensor      # Shape [L, 14] - Occupancy
     }
     ```
   - Coordinate order (14 atoms):
     ```
     0-3: Backbone atoms (N, CA, C, O)
     4-13: Side chain atoms (specific to each amino acid)
     ```

2. **PDBID.pt** (e.g., 1l3a.pt)
   - Contains structure metadata and assembly information
   - Located at: `pdb/<prefix>/<pdbid>.pt`
   - Fields:
     ```python
     {
         'method': str,                 # Experimental method (X-ray, NMR, etc.)
         'date': str,                   # Deposition date
         'resolution': float,           # Structure resolution (Å)
         'chains': List[str],           # List of chain IDs
         'tm': torch.Tensor,            # Shape [N, N, 3] - Chain similarity matrix
         'asmb_ids': List[str],         # Biounit IDs
         'asmb_details': List[str],     # Assembly identification details
         'asmb_method': List[str],      # Assembly method
         'asmb_chains': List[str],      # Chains in each biounit
         'asmb_xformN': torch.Tensor    # Shape [n, 4, 4] - Transform matrices
     }
     ```

## Reading PDB Files

Here's a detailed example of how to read and analyze PDB files:

```python
import torch
import numpy as np
import os
from typing import Dict, Any

def read_chain_file(file_path: str) -> Dict[str, Any]:
    """Read a PDBID_CHAINID.pt file and return its contents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Chain file not found: {file_path}")
    
    data = torch.load(file_path)
    
    # Validate expected fields
    expected_fields = {'seq', 'xyz', 'mask', 'bfac', 'occ'}
    missing_fields = expected_fields - set(data.keys())
    if missing_fields:
        print(f"Warning: Missing fields in chain file: {missing_fields}")
    
    return data

def read_pdb_file(file_path: str) -> Dict[str, Any]:
    """Read a PDBID.pt file and return its contents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDB file not found: {file_path}")
    
    data = torch.load(file_path)
    
    # Validate expected fields
    expected_fields = {
        'method', 'date', 'resolution', 'chains', 'tm',
        'asmb_ids', 'asmb_details', 'asmb_method', 'asmb_chains'
    }
    missing_fields = expected_fields - set(data.keys())
    if missing_fields:
        print(f"Warning: Missing fields in PDB file: {missing_fields}")
    
    return data

def analyze_structure(pdb_id: str, chain_id: str, data_dir: str) -> None:
    """Analyze a protein structure and its metadata."""
    # Construct file paths
    prefix = pdb_id[1:3].lower()
    chain_file = f"{data_dir}/pdb/{prefix}/{pdb_id}_{chain_id}.pt"
    pdb_file = f"{data_dir}/pdb/{prefix}/{pdb_id}.pt"
    
    # Read files
    chain_data = read_chain_file(chain_file)
    pdb_data = read_pdb_file(pdb_file)
    
    # Print chain information
    print(f"=== Chain {pdb_id}_{chain_id} Analysis ===")
    print(f"Sequence length: {len(chain_data['seq'])}")
    print(f"Sequence: {chain_data['seq']}")
    print(f"\nCoordinate statistics:")
    print(f"  Mean position: {chain_data['xyz'].mean(dim=(0,1))}")
    print(f"  Position std: {chain_data['xyz'].std(dim=(0,1))}")
    
    # Print structure information
    print(f"\n=== Structure {pdb_id} Analysis ===")
    print(f"Method: {pdb_data['method']}")
    print(f"Resolution: {pdb_data['resolution']:.2f} Å")
    print(f"Deposition date: {pdb_data['date']}")
    print(f"Number of chains: {len(pdb_data['chains'])}")
    
    # Print assembly information
    print(f"\nAssembly Information:")
    for i, (asmb_id, details, method, chains) in enumerate(zip(
        pdb_data['asmb_ids'],
        pdb_data['asmb_details'],
        pdb_data['asmb_method'],
        pdb_data['asmb_chains']
    )):
        print(f"\nAssembly {i+1}:")
        print(f"  ID: {asmb_id}")
        print(f"  Details: {details}")
        print(f"  Method: {method}")
        print(f"  Chains: {chains}")

# Example usage
if __name__ == "__main__":
    pdb_id = "1l3a"
    chain_id = "A"
    data_dir = "data/pdb_2021aug02_sample"
    analyze_structure(pdb_id, chain_id, data_dir)
```


Example output:
```
=== Chain 1l3a_A Analysis ===
Sequence length: 164
Sequence: MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL

Coordinate statistics:
  Mean position: tensor([ 0.1234, -0.0456,  0.7890])
  Position std: tensor([14.2345, 15.6789, 13.8901])

=== Structure 1l3a Analysis ===
Method: X-RAY DIFFRACTION
Resolution: 1.75 Å
Deposition date: 2002-08-15
Number of chains: 1

Assembly Information:
Assembly 1:
  ID: 1
  Details: author_defined_assembly
  Method: author
  Chains: A
```

## Training

To train the model:

```bash
python scripts/train.py \
    --path_for_training_data data/ \
    --path_for_outputs outputs/ \
    --hidden_dim 128 \
    --num_encoder_layers 3 \
    --num_decoder_layers 3 \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --max_epochs 100 \
    --use_cuda \
    --use_amp
```

### Key Arguments

- `--path_for_training_data`: Path to training data directory
- `--path_for_outputs`: Path to output directory
- `--hidden_dim`: Hidden dimension size (default: 128)
- `--num_encoder_layers`: Number of encoder layers (default: 3)
- `--num_decoder_layers`: Number of decoder layers (default: 3)
- `--batch_size`: Batch size (default: 1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--max_epochs`: Maximum number of epochs (default: 100)
- `--use_cuda`: Use GPU if available
- `--use_amp`: Use automatic mixed precision training

For a full list of arguments, run:
```bash
python scripts/train.py --help
```

## Model Architecture

The model consists of:
- Message passing neural network encoder
- Attention-based decoder
- Position-wise feed-forward networks
- Layer normalization and residual connections

## Features

- PyTorch Lightning implementation for better code organization and reduced boilerplate
- Automatic mixed precision training support
- TensorBoard logging
- Gradient clipping
- Learning rate scheduling
- Checkpoint saving and loading
- Multi-GPU support (through PyTorch Lightning)

## License

This project is licensed under the same terms as the original ProteinMPNN implementation. 
