import torch

def accuracy(src, target):
    """
        return num of correct predict
    """
    pred = torch.argmax(src, dim=1)
    corrects = (pred == target).sum().item()
    return corrects
    