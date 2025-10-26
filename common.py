import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def to_device(x, device):
    return x.to(device, non_blocking=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
