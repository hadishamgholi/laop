import torchvision
from torchvision import transforms
import torch
from torch import nn
import config as con
import random
import numpy as np

def leng(generator):
    return sum(1 for _ in generator)

def get_prms_rqr_grd(model):
    return list(filter(lambda p: p.requires_grad, model.parameters()))


def get_model_layers(model, return_layer_count=False):
    l = [module for module in model.modules() if (
        type(module) != nn.Sequential
        and type(module) != type(model)
        and leng(module.parameters()) > 0)]

    if return_layer_count:
        return l, len(l)
    else:
        return l


def accuracy(src, target):
    """
        return num of correct predict
    """
    pred = torch.argmax(src, dim=1)
    corrects = (pred == target).sum().item()
    return corrects
    
def set_random_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_cifar10_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=con.batch_size,
                                            shuffle=True, num_workers=con.num_worker)


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=con.batch_size,
                                            shuffle=False, num_workers=con.num_worker)

    yield trainset, trainloader, testset, testloader