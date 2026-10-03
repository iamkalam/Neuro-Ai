"""
Synthetic motion-detection dataset.

Generates sinusoidal gratings moving in one of four cardinal directions
(up / down / left / right). Each input neuron is assigned a fixed 2D
position on a virtual retinal grid; the grating intensity at that
position becomes the neuron's feature value.

Returns per-sample tensors of shape:
    x_input : [num_input_nodes]   — light intensity per input neuron
    y       : scalar in {0,1,2,3} — motion direction class
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# -----------------------------
# Configuration
# -----------------------------
DIRECTIONS = {
    "right": 0,   # grating moves +x
    "left":  1,   # grating moves -x
    "up":    2,   # grating moves +y  (visual up)
    "down":  3,   # grating moves -y
}

# Retinal grid on which input neurons are laid out
GRID_SIZE = 64          # 64 x 64 virtual retina
GRATING_WAVELENGTH = 8.0   # spatial period of the sinusoid (in grid units)
GRATING_AMPLITUDE  = 1.0
NOISE_STD          = 0.05


class MotionGratingDataset(Dataset):
    """
    Parameters
    ----------
    input_node_ids : list[int]
        Root IDs of the input neurons (photoreceptors + lamina).
        Determines how many input nodes exist and their order.

    num_samples : int
        Number of samples to generate.

    seed : int
        Random seed for reproducibility.
        Input positions are drawn once and fixed for the whole dataset.

    fixed_positions : np.ndarray or None
        If provided, reuses these node positions instead of drawing new
        ones. Used to keep train/test positions identical.
    """

    def __init__(
        self,
        input_node_ids,
        num_samples=5000,
        seed=0,
        fixed_positions=None,
    ):
        self.num_inputs = len(input_node_ids)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)

        # ---- Assign each input neuron a fixed 2D position on the retina ----
        if fixed_positions is not None:
            self.positions = fixed_positions
        else:
            # Uniform random positions on a GRID_SIZE x GRID_SIZE retina
            self.positions = self.rng.uniform(
                0.0, GRID_SIZE, size=(self.num_inputs, 2)
            ).astype(np.float32)

        # ---- Pre-generate all samples ----
        self.frames, self.labels = self._generate_all()

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    def _render_grating(self, direction, phase):
        """
        Render a sinusoidal grating of the given direction and phase
        sampled at self.positions.

        direction : str in {'right','left','up','down'}
        phase     : float in [0, 2*pi)
        Returns  : [num_inputs] float32 intensity in [0, 1]
        """
        x = self.positions[:, 0]
        y = self.positions[:, 1]

        # Spatial coordinate along the direction of motion
        if direction == "right":
            coord = x
        elif direction == "left":
            coord = -x
        elif direction == "up":
            coord = y
        elif direction == "down":
            coord = -y
        else:
            raise ValueError(f"Unknown direction: {direction}")

        k = 2.0 * np.pi / GRATING_WAVELENGTH
        intensity = GRATING_AMPLITUDE * np.sin(k * coord + phase)

        # Normalize to [0, 1] and add small noise
        intensity = 0.5 * (intensity + 1.0)
        intensity += self.rng.normal(0.0, NOISE_STD, size=intensity.shape)

        return np.clip(intensity, 0.0, 1.0).astype(np.float32)

    def _generate_all(self):
        frames = np.zeros((self.num_samples, self.num_inputs), dtype=np.float32)
        labels = np.zeros((self.num_samples,), dtype=np.int64)

        direction_names = list(DIRECTIONS.keys())
        for i in range(self.num_samples):
            d_name = direction_names[i % len(direction_names)]
            phase  = self.rng.uniform(0.0, 2.0 * np.pi)

            frames[i] = self._render_grating(d_name, phase)
            labels[i] = DIRECTIONS[d_name]

        return torch.from_numpy(frames), torch.from_numpy(labels)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.frames[idx], self.labels[idx]


# -----------------------------
# Smoke test
# -----------------------------
if __name__ == "__main__":
    # Fake input IDs just for the smoke test — real run uses the .pt graph
    fake_input_ids = list(range(4466))

    ds = MotionGratingDataset(fake_input_ids, num_samples=32, seed=42)
    x0, y0 = ds[0]

    print(f"Number of input nodes: {ds.num_inputs}")
    print(f"Number of samples:     {len(ds)}")
    print(f"Sample x shape:        {x0.shape}   dtype={x0.dtype}")
    print(f"Sample x range:        [{x0.min():.3f}, {x0.max():.3f}]")
    print(f"Sample y:              {y0.item()}")

    # Print label distribution
    import collections
    counts = collections.Counter(ds.labels.tolist())
    print(f"Label distribution:    {dict(counts)}")