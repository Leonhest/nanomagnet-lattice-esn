import torch

def spectral_radius(w):
    return torch.max(torch.abs(torch.linalg.eigvals(w))).item()