"""Convert a trained PyTorch checkpoint to a PyTorch-free aira.npz.

Run this where torch is installed (Colab / laptop), NOT in Termux:
    python export_numpy.py aira_ai_best.pt aira.npz
"""
import json
import sys

import numpy as np
import torch

src = sys.argv[1] if len(sys.argv) > 1 else "aira_ai_best.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else "aira.npz"

ck = torch.load(src, map_location="cpu")
weights = {k: v.float().numpy() for k, v in ck["model"].items() if k != "lm_head.weight"}  # tied
meta = json.dumps({"config": ck["config"], "chars": ck["chars"]})
np.savez_compressed(dst, _meta=np.array(meta), **weights)
print(f"✅ Saved {dst}  ({sum(w.size for w in weights.values()):,} weights)")
