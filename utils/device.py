import torch

def default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'