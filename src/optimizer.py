from adamp import AdamP
import torch.optim as optim

def get_sgd_optimizer(model):
    return optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
def get_adam_optimizer(model):
    return AdamP(model.parameters(), lr=1e-3) 