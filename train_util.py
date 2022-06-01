import torch 
import pickle
from torch import is_distributed, nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from metric_util import *

import sys
#sys.path.append('./PyGCL')
import GCL.losses as L
from GCL.models import DualBranchContrast
from GCL.eval import get_split, SVMEvaluator

def train(encoder_model, contrast_model, dataloader, optimizer, conditional=False, debias=False, adversarial=False, **kwargs):
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = next(encoder_model.parameters()).device
    #encoder_model.to(device)
    encoder_model.train()
    total_loss = 0.0
    total_conditional_loss = 0.0
    total_debias_loss = 0.0
    total_adv_loss = 0.0
    for batch_idx, data_batch in enumerate(dataloader):
        #print('batch_idx: ', batch_idx)
        x, ys = data_batch # x.shape: B x f, y.shape: B, s.shape: B x f
        s = ys[1]

        x = x.to(device)
        s = s.to(device)
        loss = 0.0
        conditional_loss = 0.0
        debias_loss = torch.tensor(0.0).to(device)
        adv_loss = torch.tensor(0.0).to(device)
        
        optimizer.zero_grad()

        # x has no s
        #print('x shape in train', x.shape)
        z1, z2 = encoder_model(x)

        if not conditional:
            conditional_loss += contrast_model(g1=z1, g2=z2)
            loss += conditional_loss
            #print('loss\n', loss)
        else:
            tempreture = kwargs['cond_temp'] if 'cond_temp' in kwargs else 1.0/400
            fair_kernel = abs_kernel(s,s,tempreture=tempreture).to(device)
            num_nodes = x.size(0)
            pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
            neg_mask = 1. - pos_mask
            normed_fair_kernel = fair_kernel/torch.sum(pos_mask*fair_kernel, dim=1, keepdim=True)
            #print('fair_kernel\n', fair_kernel)
            #print('normed_fair_kernel\n', normed_fair_kernel)
            #z1, z2 = encoder_model(data)

            conditional_loss += contrast_model(g1=z1, g2=z2, fair_kernel=normed_fair_kernel)
            #print('conditional_loss\n', conditional_loss)
            loss += conditional_loss
            #fair_sim = torch.sigmoid((L.infonce._similarity(g,g)-0.0) * beta)
                 
        if debias:
            tempreture = kwargs['debias_temp'] if 'debias_temp' in kwargs else 1.0/80
            debias_ratio = kwargs['debias_ratio'] if 'debias_ratio' in kwargs else 5.0
            # <x,s|z>
            z = encoder_model.main_encoder.encode_project(x)
            embed_s = encoder_model.sens_encoder.encode_project(s.float()) # age is int, transfer to float
            #print('embed_s.shape: ', embed_s.shape)
            
            # |g fair_kernel
            debias_kernel = cos_kernel(z, z, tempreture).to(device)
            num_nodes = x.size(0)
            pos_mask = torch.eye(num_nodes, dtype=torch.float32, device=device)
            neg_mask = 1. - pos_mask
            normed_debias_kernel = debias_kernel/torch.sum(pos_mask*debias_kernel, dim=1, keepdim=True)
            #print('normed_debias_kernel\n',normed_debias_kernel)
            
            debias_loss += contrast_model(g1=z, g2=embed_s, fair_kernel = normed_debias_kernel)
            loss += debias_ratio * debias_loss
            #print('debias_loss\n',debias_loss)
            #fair_matrix = abs_kernel(sens_attribute,sens_attribute).to(device)
            #fair_kernel = cos_kernel().to(device)

        # adversarial debias, mlp_adv in encoder_model     
        if adversarial:
            # adversarial prediction for sens
            z = encoder_model.main_encoder.encode_project(x)
            # predict sens
            s_pred = encoder_model.adv_model.forward(z)
            # avd loss
            # if sens is discrete/continuous
            is_discrete = s_pred.shape[1] != 1
            output_fn = nn.LogSoftmax(dim=-1) if is_discrete else nn.Sigmoid()
            criterion = nn.NLLLoss() if is_discrete else nn.MSELoss()
            s_pred = s_pred if is_discrete else s_pred.flatten()
            s = s if is_discrete else s.flatten()

            adv_loss += 0.1 * criterion(output_fn(s_pred), s)
            loss += -adv_loss
            
        #print(loss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_conditional_loss += conditional_loss.item()
        total_debias_loss += debias_loss.item()
        total_adv_loss += adv_loss.item()

    return {'loss': total_loss/len(dataloader), 
            'conditional_loss':total_conditional_loss/len(dataloader), 
            'debias_loss': total_debias_loss/len(dataloader),
            'adv_loss': total_adv_loss/len(dataloader)}
        
    
#evaluator = None
def test(encoder_model, dataloader, evaluator=None):
    device = next(encoder_model.parameters()).device
    encoder_model.eval()
    z_list = []
    y_list = []
    #dataloader = DataLoader(dataset, batch_size=10000s, shuffle=True, num_workers=4)
    with torch.no_grad():
        for data_batch in dataloader:
            x, ys = data_batch
            y = ys[0]
            x = x.to(device)
            y = y.to(device)
            z = encoder_model.main_encoder.forward(x)
            z_list.append(z)
            y_list.append(y)
        z = torch.cat(z_list, dim=0)
        y = torch.cat(y_list, dim=0)
    #print(x.shape)
    #print(y.shape)
    split = get_split(num_samples=z.size()[0], train_ratio=0.7, test_ratio=0.15)
    evaluator = SVMEvaluator(linear=True) if evaluator is None else evaluator
    #print(evaluator)
    result = evaluator(z, y, split)
    
    return result, evaluator