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

# Read PDBID_CHAINID.pt file (e.g., 2l35_A.pt)
chain_file = "./pdb_2021aug02_sample/pdb/l3/2l35_A.pt"
chain_data = torch.load(chain_file)

print("=== Contents of PDBID_CHAINID.pt ===")
print("\nChain Data:")
for key, value in chain_data.items():
    print_tensor_info(key, value)
    
# Expected output for chain data:
# - seq: amino acid sequence (string)
# - xyz: atomic coordinates [L,14,3]
# - mask: boolean mask [L,14]
# - bfac: temperature factors [L,14]
# - occ: occupancy [L,14]

# Read PDBID.pt file (e.g., 2l35.pt)
pdb_file = "./pdb_2021aug02_sample/pdb/l3/2l35.pt"
pdb_data = torch.load(pdb_file)

print("\n=== Contents of PDBID.pt ===")
print("\nPDB Metadata:")
for key, value in pdb_data.items():
    if key.startswith('asmb_xform'):
        print(f"\n{key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Type: {value.dtype}")
        print("  First transform matrix:")
        print(value[0])
    else:
        print_tensor_info(key, value)

# Expected output for PDB data:
# - method: experimental method (str)
# - date: deposition date (str)
# - resolution: resolution (float)
# - chains: list of CHAINIDs
# - tm: pairwise similarity matrix [num_chains,num_chains,3]
# - asmb_ids: biounit IDs
# - asmb_details: assembly identification method
# - asmb_method: assembly method
# - asmb_chains: chains in each biounit
# - asmb_xformN: transformation matrices [n,4,4]