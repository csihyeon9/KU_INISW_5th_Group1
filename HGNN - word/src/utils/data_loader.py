import torch

def load_data(data_path):
    data = torch.load(data_path)
    X, H, labels = data['X'], data['H'], data['labels']

    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X and labels size mismatch: {X.shape[0]} != {labels.shape[0]}")
    if H.shape[0] != X.shape[0]:
        raise ValueError(f"H and X size mismatch: {H.shape[0]} != {X.shape[0]}")

    return X, H, labels
