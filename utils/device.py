def default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return 'cudaa'
    else:
        return 'cpu'
device = default_device()
