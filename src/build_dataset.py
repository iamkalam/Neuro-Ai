"""
Driver script: load the connectome graph, extract the real input neuron IDs,
and build train / test MotionGratingDataset instances.
"""

import torch
from src.synthetic_motion import MotionGratingDataset


def build_datasets(graph_path="data/processed/maol_motion_subgraph.pt"):
    # ---- Load graph ----
    data = torch.load(graph_path, weights_only=False)
    print(f"Graph loaded: {data.num_nodes} nodes, {data.num_edges} edges")

    # ---- Extract real input neuron Root IDs from the graph ----
    input_root_ids = [
        data.node_ids[i]
        for i, is_input in enumerate(data.input_mask.tolist())
        if is_input
    ]
    print(f"Input neurons found: {len(input_root_ids)}")

    # ---- Build train dataset (creates positions) ----
    train_ds = MotionGratingDataset(
        input_root_ids, num_samples=5000, seed=0
    )

    # ---- Build test dataset (reuses train positions) ----
    test_ds = MotionGratingDataset(
        input_root_ids, num_samples=1000, seed=1,
        fixed_positions=train_ds.positions,
    )

    print(f"Train samples: {len(train_ds)}  |  Test samples: {len(test_ds)}")
    print(f"Input dimension: {train_ds.num_inputs}")

    return data, train_ds, test_ds


if __name__ == "__main__":
    build_datasets()