# Neuro-Ai
# Neuro-AI: Bio-Inspired Sparse Graph Neural Networks from the Fruit Fly Connectome

A final year project exploring whether the synaptic wiring of the *Drosophila* visual system provides a useful inductive bias for training sparse, parameter-efficient Graph Neural Networks.

---

## 📖 Overview

Modern deep learning models are dense, massive, and energy-hungry. Meanwhile, a fruit fly performs complex behaviors—flight, navigation, motion detection—using only ~130,000 neurons and ~20 watts of power. This gap suggests that biological wiring contains structural priors that artificial networks lack.

**This project asks:** *Does the actual synaptic connectivity of the fly's optic lobe provide a useful inductive bias for training sparse neural networks?*

We extract a functional sub-network from the FlyWire connectome, convert it into a Graph Neural Network, and benchmark it against dense and random-sparse baselines on a synthetic motion-detection task.

---

## 🎯 Research Question

> Given the same task and comparable parameter budgets, does a GNN wired according to the fruit fly's biological connectome outperform (a) a fully-connected dense baseline and (b) a randomly-wired sparse baseline of equal density?

If the connectome model beats both baselines, it demonstrates that **biological wiring itself carries useful structure**—not just sparsity in general.

---

## 🧠 Background

### Biological Motivation

The fly's **optic lobe** contains the canonical motion-detection circuit:

- **Photoreceptors (R1–R8)** capture light intensity
- **Lamina neurons (L1–L5)** perform initial gain control
- **Medulla neurons (Mi1, Tm1, Tm2, Tm3, Tm9, ...)** compute temporal correlations
- **T4 and T5 cells** are the first direction-selective neurons, with four subtypes encoding the four cardinal directions

The T4/T5 circuit is one of the most well-characterized motion-detection systems in neuroscience, making it an ideal target for a connectome-constrained model.

### Machine Learning Motivation

Graph Neural Networks (GNNs) are a natural fit for connectome data:

- Neurons → **nodes**
- Synapses → **edges**
- Synapse counts → **edge weights**
- Cell types → **node features**

The resulting graph is inherently sparse (~124 connections per neuron in the optic lobe), making it a natural testbed for parameter-efficient learning.

---

## 📊 Dataset

**MAOL v1.1 — Male Adult Fly Right Optic Lobe**

| Property | Value |
|---|---|
| Source | FlyWire / Codex |
| Neurons | 52,445 |
| Connections | 6,484,936 |
| Region | Right optic lobe |
| Sex | Male |

### Files Used

1. **`neurons.csv.gz`** — Per-neuron annotations
   - `Root ID` — unique neuron identifier
   - `Primary Cell Type` — cell type label (e.g., `T4a`, `T5b`)
   - `Flow` — sensory / intrinsic / motor classification
   - `Predicted NT type` — neurotransmitter prediction (used for excitatory/inhibitory sign)

2. **`connections.csv.gz`** — Filtered synaptic connectivity table
   - `pre` / `post` — presynaptic and postsynaptic Root IDs
   - `neuropil` — brain region of the synapse
   - `syn_count` — number of synapses (aggregated)
   - `nt_type` — predicted neurotransmitter for the connection

> **Threshold note:** The connectivity table excludes connections with fewer than 3 total synapses and autapses, reducing spurious edges from automated synapse detection.

---

## 🏗️ Methodology

### 1. Sub-Network Extraction

From the full 52k-neuron optic lobe graph, we extract a biologically meaningful sub-network:

1. Identify all **T4/T5 neurons** from cell-type annotations → these become **readout nodes**
2. Include their **direct presynaptic partners** (1-hop neighborhood)
3. Include **photoreceptors and lamina neurons** → these become **input nodes**
4. Filter the edge list to keep only edges where both `pre` and `post` are in this set

This reduces the graph from ~52,000 nodes to a few thousand—tractable for training on a single GPU.

### 2. Graph Construction (PyTorch Geometric)

```python
from torch_geometric.data import Data
import torch

data = Data(
    x=node_features,           # one-hot cell type encoding
    edge_index=edge_index,     # 2 x E tensor of (pre, post) index pairs
    edge_attr=edge_weights,    # normalized synapse counts
)
data.readout_mask = ...        # boolean mask over T4/T5 nodes
data.input_mask = ...          # boolean mask over input neurons



neuro-ai-fyp/
├── data/
│   ├── raw/
│   │   ├── neurons.csv.gz
│   │   └── connections.csv.gz
│   └── processed/
│       └── maol_motion_subgraph.pt
├── src/
│   ├── load_data.py           # load and inspect raw files
│   ├── build_graph.py         # filter subnetwork, build PyG Data
│   ├── synthetic_motion.py    # generate moving grating dataset
│   ├── model.py               # GraphSAGE motion classifier
│   ├── baselines.py           # dense + random-sparse baselines
│   └── train.py               # training and evaluation loop
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_build_graph.ipynb
│   └── 03_train_and_evaluate.ipynb
├── results/
│   ├── figures/
│   └── metrics.csv
├── requirements.txt
└── README.md




🚀 Getting Started
1. Clone the Repository
bash
git clone https://github.com/<your-username>/neuro-ai-fyp.git
cd neuro-ai-fyp
2. Install Dependencies
bash
pip install -r requirements.txt
requirements.txt:

text
torch>=2.0
torch-geometric>=2.4
pandas
numpy
matplotlib
networkx
scikit-learn
3. Download the Data
Go to https://codex.flywire.ai/api/download

Select MAOL v1.1 (OL/R) — Male Adult Fly Right Optic Lobe

Download:

neurons.csv.gz (annotations)

connections.csv.gz (filtered connectivity, 3+ synapse threshold)

Place both files in data/raw/

4. Build the Sub-Network Graph
bash
python src/build_graph.py
This produces data/processed/maol_motion_subgraph.pt.


📚 References
FlyWire Consortium — Neuronal wiring diagram of an adult brain, Nature, 2024

flyGNN — Whole-brain connectome GNN for embodied locomotion control, 2024

Watts & Webb — Connectome-constrained larval olfactory model with 449 trainable parameters

Nature (2024) — Connectome-constrained motion detection network with 64 cell types

PyTorch Geometric — https://pytorch-geometric.readthedocs.io

FlyWire Codex — https://codex.flywire.ai

 Author
Abdul Kalam
BS Computer Science 
FAST NUCES

📄 License
This project uses publicly available FlyWire/Codex data. Code is released under the MIT License. See LICENSE for details.

🙏 Acknowledgments
FlyWire Consortium and the Codex team for open connectome data

PyTorch Geometric developers for the GNN framework

The neuroscience community for decades of work characterizing the fly visual system