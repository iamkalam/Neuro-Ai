import torch
data = torch.load("maol_motion_subgraph.pt", weights_only=False)
print(data)
print(f"Nodes: {data.num_nodes}")
print(f"Edges: {data.num_edges}")
print(f"Readout: {data.readout_mask.sum().item()}")
print(f"Input:   {data.input_mask.sum().item()}")