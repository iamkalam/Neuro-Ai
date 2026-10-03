import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

# ---------- 1. Load data ----------
neurons = pd.read_csv("data/raw/neurons.csv.gz", compression="gzip")
conns   = pd.read_csv("data/raw/connections_princeton.csv.gz", compression="gzip")

pre_col  = "pre_root_id"
post_col = "post_root_id"
syn_col  = "syn_count"
ct_col   = "Primary Cell Type"
id_col   = "Root ID"

# ---------- 2. Aggregate duplicate (pre, post) rows across neuropils ----------
edges = (
    conns.groupby([pre_col, post_col], as_index=False)[syn_col]
    .sum()
)
print(f"Aggregated edges: {len(edges)}")

# ---------- 3. Identify biological node sets ----------
# Readout: all T4 and T5 subtypes
t4t5_mask = neurons[ct_col].astype(str).str.match(r"^T4|^T5", na=False)
t4t5_ids  = set(neurons.loc[t4t5_mask, id_col])
print(f"T4/T5 neurons: {len(t4t5_ids)}")

# Inputs: photoreceptors R1–R8 and lamina L1–L5
input_pattern = r"^R[1-8]$|^L[1-5]$"
input_mask    = neurons[ct_col].astype(str).str.match(input_pattern, na=False)
input_ids     = set(neurons.loc[input_mask, id_col])
print(f"Input neurons (R/L): {len(input_ids)}")

# ---------- 4. Filter edges to the STRICT motion sub-network ----------
# Keep only canonical motion-pathway interneurons as upstream partners
motion_types = {
    # Medulla intrinsic neurons (motion computation)
    "Mi1", "Mi4", "Mi9",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Tm20",
    # Medulla-to-lobula projection neurons
    "T2", "T2a", "T3",
    # Lobula / lobula plate cells presynaptic to T4/T5
    "C2", "C3",
    # Lamina monopolar cells
    "L1", "L2", "L3", "L4", "L5",
    # Photoreceptors
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
}

# Direct presynaptic partners of T4/T5
t4t5_edges    = edges[edges[post_col].isin(t4t5_ids)]
raw_upstream  = set(t4t5_edges[pre_col])

# Restrict upstream to only canonical motion-pathway cell types
upstream_mask = neurons[ct_col].astype(str).isin(motion_types)
typed_upstream = set(neurons.loc[upstream_mask, id_col])
upstream_ids = raw_upstream & typed_upstream

print(f"Raw upstream partners: {len(raw_upstream)}")
print(f"Filtered upstream (canonical types): {len(upstream_ids)}")

# Build the keep set
keep_ids = t4t5_ids | upstream_ids | input_ids
print(f"Total keep set: {len(keep_ids)}")

# Filter edges to only those between kept nodes
filtered = edges[
    edges[pre_col].isin(keep_ids) & edges[post_col].isin(keep_ids)
].copy()

print(f"Filtered edges: {len(filtered)}")
print(f"Filtered nodes: {len(set(filtered[pre_col]) | set(filtered[post_col]))}")

# ---------- 5. Map Root IDs → 0..N-1 indices ----------
unique_ids  = sorted(set(filtered[pre_col]) | set(filtered[post_col]))
node_to_idx = {rid: i for i, rid in enumerate(unique_ids)}
N = len(unique_ids)
print(f"Node count: {N}")

# ---------- 6. Edge tensors ----------
src = filtered[pre_col].map(node_to_idx).values
dst = filtered[post_col].map(node_to_idx).values

edge_index = torch.from_numpy(np.vstack([src, dst]).astype(np.int64))

w = filtered[syn_col].values.astype("float32")
w = w / w.max()
edge_attr = torch.from_numpy(w).unsqueeze(-1)

# ---------- 7. Node features (one-hot of cell type) ----------
ct_series = (
    neurons.set_index(id_col)
           .loc[unique_ids, ct_col]
           .fillna("unknown")
           .astype(str)
)
ct_onehot = pd.get_dummies(ct_series).values.astype("float32")
x = torch.from_numpy(ct_onehot)
print(f"Node feature dim: {x.shape[1]}")

# ---------- 8. Boolean masks for input / readout ----------
readout_mask = torch.tensor(
    [rid in t4t5_ids for rid in unique_ids], dtype=torch.bool
)
input_mask_t = torch.tensor(
    [rid in input_ids for rid in unique_ids], dtype=torch.bool
)

# ---------- 9. Build PyG Data object ----------
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
data.readout_mask = readout_mask
data.input_mask   = input_mask_t
data.node_ids     = unique_ids

print("\n=== FINAL GRAPH ===")
print(data)
print(f"Readout nodes (T4/T5): {readout_mask.sum().item()}")
print(f"Input nodes (R/L):     {input_mask_t.sum().item()}")

# ---------- 10. Save ----------
torch.save(data, "maol_motion_subgraph_12k.pt")
print("\nSaved to maol_motion_subgraph_12k.pt")