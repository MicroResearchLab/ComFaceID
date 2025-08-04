from torch.nn.modules.activation import SELU, ReLU, Sigmoid
from torch.nn.modules.linear import Linear
from torch.utils.data import dataset
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import math
import torch




def weight_init(model):
    for n, m in model.named_children():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)




class DS_Net(nn.Module):
    def __init__(self, base_features_size,output_size):
        super(DS_Net, self).__init__()
        self.base_features_size = base_features_size
        self.encoder = nn.Sequential(
            Linear(base_features_size, 2048),
            nn.ReLU(), 
            nn.LayerNorm(2048),
            Linear(2048, 1024),
            nn.ReLU(), 
            Linear(1024, output_size),
        )

        self.weight_init()
        
    def forward(self, x):
        res= self.encoder(x)
        return res

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, (torch.nn.Linear)):
                nn.init.xavier_uniform_(m.weight)