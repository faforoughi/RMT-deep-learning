import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self,input_dim=64,width=256,num_classes=10):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,width),nn.ReLU(),
                               nn.Linear(width,width),nn.ReLU(),
                               nn.Linear(width,num_classes))
    def forward(self,x): return self.net(x)

def weight_matrices(model):
    return {name:m.weight.detach().cpu().numpy().copy()
            for name,m in model.named_modules() if isinstance(m,nn.Linear)}
