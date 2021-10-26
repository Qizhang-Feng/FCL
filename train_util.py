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

def train(encoder_model, contrast_model, dataloader, optimizer, train=True, conditional=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder_model.to(device)
    encoder_model.train()
    total_loss = 0.0
    for batch_idx, data_batch in enumerate(dataloader):
        #print('batch_idx: ', batch_idx)
        data, label = data_batch
        if not conditional:
            data = data.to(device)
            optimizer.zero_grad()

            g1, g2 = encoder_model(data)
            #print(g1.shape)
            #print(g2.shape)

            loss = contrast_model(g1=g1, g2=g2)
        else:
            
            data_0 = data[np.where(data[:, 40:41] == 0)[0]].to(device)
            data_0[:, 40:41] = 0 # remove sa
            data_1 = data[np.where(data[:, 40:41] == 1)[0]].to(device)
            data_1[:, 40:41] = 0 # remove sa
            #print('data_0:', data_0.shape)
            #print('data_1:', data_1.shape)
            ratio_0 = data_0.shape[0] / data.shape[0]
            ratio_1 = data_1.shape[0] / data.shape[0]
            optimizer.zero_grad()
            g1, g2 = encoder_model(data_0)
            loss = ratio_0 * contrast_model(g1=g1, g2=g2)
            
            g1, g2 = encoder_model(data_1)
            loss += ratio_1 * contrast_model(g1=g1, g2=g2)
            
        #print(loss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss
        
    
#evaluator = None
def test(encoder_model, dataloader, evaluator=None):
    encoder_model.eval()
    x = []
    y = []
    for data_batch in dataloader:
        data, label = data_batch
        data = data.to(device)
        label = label.to(device)
        g = encoder_model.encoder(data)
        x.append(g)
        y.append(label)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    print(x.shape)
    print(y.shape)
    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    evaluator = SVMEvaluator(linear=True) if evaluator is None else evaluator
    result = evaluator(x, y, split)
    
    return result, evaluator