"""Load the independently trained structures 1, 2, 3 and 4."""
import json
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def load_config(structure_id):
    """Return blocks/kwargs/parameters and weighted edges as a dictionary."""
    if structure_id not in (1, 2, 3, 4):
        raise ValueError('structure_id must be 1, 2, 3 or 4')
    path = CONFIG_DIR/f'structure_{structure_id}.json'
    return json.loads(path.read_text(encoding='utf-8'))


def load_structure(structure_id, seed=0, device='cpu', dtype=None):
    """Build executable blocks and adjacency matrix using saved parameters."""
    import sys
    import torch
    sys.path.insert(0, str(CONFIG_DIR.parent))
    from NeuroEA_GEA_torch.entropy_probe import build_structure
    from NeuroEA_GEA_torch.utils.rng import NumpyTorchRNG
    dtype = torch.float64 if dtype is None else dtype
    blocks, graph = build_structure(load_config(structure_id),
                                    NumpyTorchRNG(seed, device=device, dtype=dtype),
                                    device=device, dtype=dtype)
    # Keep runtime RNG independent of random values consumed by constructors.
    runtime_rng = NumpyTorchRNG(seed, device=device, dtype=dtype)
    for block in blocks:
        block.rng = runtime_rng
    return blocks, graph


if __name__ == '__main__':
    for identifier in (1, 2, 3, 4):
        blocks, graph = load_structure(identifier)
        print(f'Structure {identifier}: {len(blocks)} blocks, '
              f'{sum(b.parameter.numel() for b in blocks)} parameters, '
              f'adjacency {graph.shape}')
