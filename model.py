import torch 
import pickle
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

import sys
sys.path.append('./PyGCL')
import GCL.losses as L
from GCL.models import DualBranchContrast
from GCL.eval import get_split, SVMEvaluator

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_conv(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    return F.dropout(x, drop_prob) * (1 - drop_prob)

class FeatureDrop(nn.Module):
    def __init__(self, drop_prob):
        super(FeatureDrop, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_feature(x, self.drop_prob)


class Encoder(nn.Module):
    def __init__(self, encoder, augmentor):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.augmentor = augmentor
        
    def forward(self, x):
        x1 = self.augmentor(x)
        x2 = self.augmentor(x)
        g1, g2 = self.encoder.encode_project(x1), self.encoder.encode_project(x2)
        return (g1, g2)
    
# data augmented before feed in MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            make_conv(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            make_conv(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )
        
        
        self.project_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def project(self, z):
        return self.project_head(z)
        
    def encode_project(self, x):
        return self.project_head(self.layers(x))