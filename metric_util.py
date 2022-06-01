import atexit
import torch.nn.functional as F
import torch
import numpy as np
from matplotlib import pyplot as plt
from data_util import get_samples

def _similarity(h1: torch.Tensor, h2: torch.Tensor):
    h1 = F.normalize(h1)
    h2 = F.normalize(h2)
    return h1 @ h2.t()

def sng_dis(Z):
    h = F.normalize(Z)
    return torch.sum(torch.exp(h@h.T) * (1 - torch.eye(h.size(0), dtype=torch.float32, device=h.device)))/(h.size(0)*(h.size(0)-1))

def dua_dis(z1, z2):
    return torch.mean(torch.exp(F.normalize(z1) @ F.normalize(z2).T))

def dis_dist(Z, n):
    h = F.normalize(Z)
    anchor = h[n,:]
    dis = h @ anchor
    return dis

def pr(predictions):
    return sum(predictions)/len(predictions) 

#@torch.no_grad()
def cos_kernel(anchor, sample, tempreture=1.0):
    anchor = anchor.float()
    sample = sample.float()
    cos_sim = _similarity(anchor, sample)/tempreture
    return torch.exp(cos_sim)

#@torch.no_grad()
def abs_kernel(anchor, sample, tempreture=1.0):
    # anchor, sample shape: B x f
    B = anchor.size(0)
    anchor = anchor.float()
    sample = sample.float()
    abs_sim = -torch.abs(anchor.repeat(1,B) - sample.repeat(1,B).T)/tempreture
    return torch.exp(abs_sim)

def sigmoid_kernel(anchor, sample, tempreture=1.0):
    anchor = anchor.float()
    sample = sample.float()
    cos_sim = _similarity(anchor, sample)/tempreture
    return torch.sigmoid(cos_sim)
    
# demographic parity for classification model
# binary class, binary sens
def cls_dp(x_0: torch.FloatTensor, x_1: torch.FloatTensor, encoder_model, classifier):
    # x_0, x_1: input from two groups
    device = next(encoder_model.parameters()).device
    x_0 = x_0.to(device)
    x_1 = x_1.to(device)
    encoder_model.eval()
    classifier.to(device)
    classifier.eval()
    
    # check classifier type
    pred_func = classifier.predict if hasattr(classifier, 'predcit') else classifier.__call__
    
    # get embedd
    g_0 = encoder_model.main_encoder.encode_project(x_0)
    g_1 = encoder_model.main_encoder.encode_project(x_1)
    
    # predict
    pred_0 = pred_func(g_0).argmax(-1).detach().cpu().numpy()
    pred_1 = pred_func(g_1).argmax(-1).detach().cpu().numpy()
    
    # pos rate
    pr_0 = np.mean(pred_0)
    pr_1 = np.mean(pred_1)

    print(len(pred_0))
    print(len(pred_1))

    print(pr_0)
    print(pr_1)
    
    
    
    dp = np.abs(pr_0 - pr_1)
    
    return dp

@torch.no_grad()
def gdp(dataset, encoder_model, classifier,x,sens, hist_num = 100, task = 'classification'):
    '''
    :para x: input data
    :para sens: sensitive feature, continuous field, shape B x f
    '''
    assert task in ['classification', 'regression']
    device = next(encoder_model.parameters()).device
    encoder_model = encoder_model.to(device)
    x = x.to(device)
    sens = (sens.numpy()) # shape B x 1
    encoder_model.eval()
    classifier.to(device)
    classifier.eval()

    # check classifier type
    pred_func = classifier.predict if hasattr(classifier, 'predcit') else classifier.__call__

    z = encoder_model.main_encoder.encode_project(x)
    pred = np.expand_dims(pred_func(z).argmax(-1).detach().cpu().numpy(), axis=1) if task == 'classification' else torch.sigmoid(pred_func(z)).detach().cpu().numpy()# shape B * 1



    sens_max = np.max(sens) + 1e-5
    sens_min = np.min(sens)

    bin_idx = np.squeeze(((sens - sens_min)/(sens_max - sens_min) * hist_num // 1).astype('int'))
    n_values = np.max(bin_idx) + 1
    one_hot_bin_idx = np.eye(n_values)[bin_idx] # shape B * b
    
    #plt.scatter(bin_idx, pred.T)
    #plt.show()


    gdp_hist = np.sum(np.abs(np.mean(pred*one_hot_bin_idx, axis=0) - np.mean(pred) * np.mean(one_hot_bin_idx, axis=0)))

    test_sol = 1.0/hist_num
    x_appro = torch.arange(test_sol, 1-test_sol, test_sol).to(device)
    KDE_FAIR = kde_fair(x_appro)
    # transfer to flat torch.tensor
    gdp_kernel =  KDE_FAIR.forward(torch.tensor(pred).flatten().float(), torch.tensor(sens).flatten(), device).detach().cpu().numpy()


    # max difference
    
    bin_num = np.sum(one_hot_bin_idx, axis=0)
    bin_sum = np.sum(pred*one_hot_bin_idx, axis=0)
    
    bin_mean_ = []
    for i in range(bin_num.shape[-1]):
        if bin_num[i] != 0.0:
            bin_mean_.append(bin_sum[i] / bin_num[i])

    gdp_max = np.max(bin_mean_) - np.min(bin_mean_)
    return gdp_hist, gdp_kernel, gdp_max


import torch
from torch import nn
from math import pi, sqrt
import numpy as np

class kde_fair:
    """
    A Gaussian KDE implemented in pytorch for the gradients to flow in pytorch optimization.
    Keep in mind that KDE are not scaling well with the number of dimensions and this implementation is not really
    optimized...
    """
    def __init__(self, x_test):
        # self.train_x = x_train
        # self.train_y = y_train
        self.x_test = x_test
    
    def forward(self, y_train, x_train, device_gpu):
        n = x_train.size()[0]
        # print(f'n={n}')
        d = 1
        bandwidth = torch.tensor((n * (d + 2) / 4.) ** (-1. / (d + 4))).to(device_gpu)
        y_train = y_train.to(device_gpu)
        x_train = x_train.to(device_gpu)


        y_hat = self.kde_regression(bandwidth, x_train, y_train)
        y_mean = torch.mean(y_train)
        pdf_values = self.pdf(bandwidth, x_train)

        DP = torch.sum(torch.abs(y_hat-y_mean) * pdf_values) / torch.sum(pdf_values)
        return DP

    def kde_regression(self, bandwidth, x_train, y_train):
        n = x_train.size()[0]
        X_repeat = self.x_test.repeat_interleave(n).reshape((-1, n))
        attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2/(bandwidth ** 2) / 2, dim=1)
        y_hat = torch.matmul(attention_weights, y_train)
        return y_hat

    def pdf(self, bandwidth, x_train):
        n = x_train.size()[0]
        # data = x.unsqueeze(-2)
        # train_x = _unsqueeze_multiple_times(self.train_x, 0, len(s))

        data = self.x_test.repeat_interleave(n).reshape((-1, n))
        train_x = x_train.unsqueeze(0)
        # print(f'data={data.shape}')
        # print(f'train_x={train_x.shape}')

        pdf_values = (torch.exp(-((data - train_x) ** 2 / (bandwidth ** 2) / 2))
                     ).mean(dim=-1) / sqrt(2 * pi) / bandwidth

        return pdf_values

# DP for regression model
def reg_dp():
    dp = None
    return dp
