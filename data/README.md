# Data Preparation Guide

This guide describes the protein structure dataset used for training RosettaFold/MPNN models.

## Download Training Data

To download the sample training dataset, run:
```bash
wget https://files.ipd.uw.edu/pub/training_sets/pdb_2021aug02_sample.tar.gz
```

## Dataset Structure

After downloading and extracting the dataset, you will find the following structure:
- `pdb_2021aug02_sample/`
  - `pdb/` - Contains protein structure files in .pt format
  - `list.csv` - Metadata about the protein structures
  - `valid_clusters.txt` - Validation set cluster information
  - `test_clusters.txt` - Test set cluster information

## File Format Details

### Chain-Specific Files (PDBID_CHAINID.pt)
Each chain is stored in a separate file with the following fields:
- `seq` - Amino acid sequence (string)
- `xyz` - Atomic coordinates [L,14,3]
- `mask` - Boolean mask [L,14]
- `bfac` - Temperature factors [L,14]
- `occ` - Occupancy [L,14] (1 for most atoms, <1 if alternative conformations present)

### PDB Entry Files (PDBID.pt)
Contains metadata and biological assembly information:
- `method` - Experimental method (str)
- `date` - Deposition date (str)
- `resolution` - Resolution (float)
- `chains` - List of CHAINIDs
- `tm` - Pairwise similarity between chains [num_chains,num_chains,3]
  - TM-score
  - Sequence identity
  - RMSD from TM-align
- `asmb_ids` - Biounit IDs as in the PDB (list of str)
- `asmb_details` - Assembly identification method (author/software)
- `asmb_method` - Assembly determination method (e.g., PISA)
- `asmb_chains` - Chains in each biounit (comma-separated CHAINIDs)
- `asmb_xformIDX` - Transformation matrices for biounits [n,4,4]
  - [n,:3,:3] - Rotation matrices
  - [n,3,:3] - Translation vectors

### Metadata File (list.csv)
Contains information about each chain:
- `CHAINID` - Chain label (PDBID_CHAINID)
- `DEPOSITION` - Deposition date
- `RESOLUTION` - Structure resolution
- `HASH` - Unique 6-digit sequence hash
- `CLUSTER` - Sequence cluster ID (30% sequence identity clusters)
- `SEQUENCE` - Reference amino acid sequence

### Cluster Files
- `valid_clusters.txt` - Clusters used for validation
- `test_clusters.txt` - Clusters used for testing

## Usage

1. Download the dataset using the wget command above
2. Extract the tar.gz file:
```bash
tar -xzf pdb_2021aug02_sample.tar.gz
```
3. The data is ready to be used with the training scripts in the `scripts/` directory

## Data Processing

The protein structures are preprocessed into PyTorch tensor format (.pt files) containing:
- Backbone atom coordinates (N, CA, C, O atoms)
- Amino acid sequences
- Chain information
- Secondary structure annotations
- Temperature factors
- Occupancy values
- Biological assembly information

## Reading the Data

Here's an example of how to read and inspect the data files using Python:

```python
import torch
import numpy as np

def print_tensor_info(name, tensor):
    """Helper function to print tensor information"""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}:")
        print(f"  Shape: {tensor.shape}")
        print(f"  Type: {tensor.dtype}")
        print(f"  First few values: {tensor[:5]}")
    else:
        print(f"{name}: {tensor}")
    print()

# Read chain-specific file (PDBID_CHAINID.pt)
chain_file = "./pdb_2021aug02_sample/pdb/l3/2l35_A.pt"
chain_data = torch.load(chain_file)

print("=== Contents of PDBID_CHAINID.pt ===")
for key, value in chain_data.items():
    print_tensor_info(key, value)

# Read PDB metadata file (PDBID.pt)
pdb_file = "./pdb_2021aug02_sample/pdb/l3/2l35.pt"
pdb_data = torch.load(pdb_file)

print("=== Contents of PDBID.pt ===")
for key, value in pdb_data.items():
    if key.startswith('asmb_xform'):
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {value.dtype}")
        print("  First transform matrix:")
        print(value[0])
    else:
        print_tensor_info(key, value)
```

This code will help you inspect:
- Chain-specific data (sequence, coordinates, masks, etc.)
- PDB metadata (experimental method, resolution, assembly information)
- Transformation matrices for biological assemblies