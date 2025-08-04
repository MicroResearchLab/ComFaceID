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




class Net(nn.Module):
    def __init__(self, base_features_size,output_size,is_Multiclass=False):
        super(Net, self).__init__()
        self.is_Multiclass = is_Multiclass
        self.base_features_size = base_features_size
        self.encoder = nn.Sequential(
            Linear(base_features_size, 512),
            nn.ReLU(), 
            nn.LayerNorm(512),
            Linear(512, output_size),
        )

        self.tail = nn.LogSoftmax(dim=1)
        self.weight_init()
        
    def forward(self, x):
        res= self.encoder(x)
        if not self.is_Multiclass:
            res = self.tail(res)
        return res

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, (torch.nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

