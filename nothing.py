# %%

from data_util import *
from metric_util import *
from train_util import *
from model import *

import torch 
import pickle
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('./PyGCL')
import GCL.losses as L
from GCL.models import DualBranchContrast
from GCL.eval import get_split, LREvaluator


# %%
config = {
    # meta config
    'dataset_name': 'celeba',
    'sens_name': 'gender',
    'conditional': False,
    'debias': False,
    'adversarial': False,
    # tunable config
    'batch_size': 256 * 2,
    'hidden_dim': 240,
    'drop_prob': 0.2,
    'cond_temp': 1.0/200,
    'debias_temp': 1.0/30,
    'debias_ratio': 4,
    'lr': 0.00005,
    'tau': 0.1,
}

# %%
# some more config setting
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

dataset_name = config['dataset_name']
sens_name = config['sens_name']
sens_num = 2 if sens_name=='gender' else 1
TASK_TYPE = 'regression' if dataset_name=='crimes' else 'classification'

# %%
# load dataset...
dataset = get_dataset(dataset_name, sens_name)
#x = dataset[:][0]
#sens = dataset[:][2]
x, sens = get_samples(dataset, num=1000)
dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

# %%
# prepare model config
#input_dim = dataset[0][0].shape[-1]
hidden_dim = config['hidden_dim'] if config['dataset_name'] != 'celeba' else 1000
sens_dim = dataset.sens_dim

# %%
# create model
main_encoder = RES()
sens_encoder = MLP(sens_dim,hidden_dim)
adv_model = Adv_sens(sens_num=sens_num, hidden_dim=hidden_dim)

aug = transforms.Compose([transforms.RandomCrop(size=RESIZE), transforms.ColorJitter(),
                         transforms.Grayscale(num_output_channels=3), transforms.RandomHorizontalFlip(),
                         transforms.RandomVerticalFlip()])#FeatureDrop(drop_prob=config['drop_prob'])

encoder_model = Encoder(main_encoder = main_encoder, augmentor = aug, sens_encoder = sens_encoder, adv_model=adv_model)
encoder_model = encoder_model.to(device)

# %%
contrast_model = DualBranchContrast(loss=L.FairInfoNCE(tau=config['tau']), mode='G2G').to(device)
optim = Adam(encoder_model.parameters(), lr=config['lr'])

performance_list = []
hist_gdp_list = []
max_gdp_list = []
kernel_gdp_list = []

epoch = 100
with tqdm(total=epoch, desc='(T)') as pbar:
    for epoch in range(1, epoch+1):
        encoder_model = encoder_model.to(device)
        loss_result = train(encoder_model = encoder_model, contrast_model=contrast_model,
                                         dataloader=dataloader, optimizer = optim,
                                         conditional=config['conditional'],debias=config['debias'], adversarial=config['adversarial'] if epoch%5==0 else False,
                                         cond_temp = config['cond_temp'],
                                         debias_temp = config['debias_temp'],
                                         debias_ratio = config['debias_ratio'])
        pbar.set_postfix({'loss': loss_result['loss'], 
                          'conditional_loss':loss_result['conditional_loss'], 
                          'debias_loss': loss_result['debias_loss'],
                          'adv_loss': loss_result['adv_loss']})
        pbar.update()
        
        if epoch % 1 == 0:
            print(loss_result)
            result, evaluator = test(encoder_model, dataloader, evaluator=LREvaluator(task=TASK_TYPE))
            classifier = result['classifier']
            
            # performance 
            performance = result['mae'] if dataset_name=='crimes' else result['auc']
            print('performance: ', performance)
            performance_list.append(performance)

            # fairness
            hist_gdp = gdp(mode='hist', task=TASK_TYPE, hist_num=1000, x = x, sens = sens, encoder_model = encoder_model, classifier = classifier)
            print('hist gdp: ', hist_gdp)
            hist_gdp_list.append(hist_gdp)
            max_gdp = gdp(mode='max', task=TASK_TYPE, hist_num=1000, x = x, sens = sens, encoder_model = encoder_model, classifier = classifier)
            print('max gdp: ', max_gdp)
            max_gdp_list.append(max_gdp)
            kernel_gdp = gdp(mode='kernel', task=TASK_TYPE, hist_num=1000, x = x, sens = sens, encoder_model = encoder_model, classifier = classifier)
            print('kernel gdp: ', kernel_gdp)
            kernel_gdp_list.append(kernel_gdp)
            #print(' auc: ', result['auc'], ' dp: ', dp)
        


