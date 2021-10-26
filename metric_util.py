import torch.nn.functional as F
import torch

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