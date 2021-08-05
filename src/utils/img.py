import torch

def denorm(x):
    x = (x+1.) / 2.
    x = torch.clamp(x, max=1.0, min=0.0)
    return x