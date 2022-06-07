import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torch_geometric.nn import GCNConv

def make_conv(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


def make_project_head(hidden_dim):
    project_head = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim)),
            #nn.ReLU(inplace=True),
            #nn.Linear(hidden_dim, hidden_dim)
        )
    return project_head

def drop_feature(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    return F.dropout(x, drop_prob) * (1 - drop_prob)

class FeatureDrop(nn.Module):
    def __init__(self, drop_prob):
        super(FeatureDrop, self).__init__()
        self.drop_prob = drop_prob

    
    def forward(self, x):
        return drop_feature(x, self.drop_prob)


class Encoder(nn.Module):
    def __init__(self, main_encoder, augmentor, **kwargs):
        super(Encoder, self).__init__()
        self.main_encoder = main_encoder
        self.augmentor = augmentor
        if 'sens_encoder' in kwargs:
            self.sens_encoder = kwargs['sens_encoder']
        if 'adv_model' in kwargs:
            self.adv_model = kwargs['adv_model']
        
    def forward(self, g):
        x,  edge_index, edge_weights= g.x, g.edge_index, g.edge_weights
        if edge_index is not None:
            g1, edge_index1, edge_weights1 = self.augmentor(x,  edge_index, edge_weights)
            g2, edge_index2, edge_weights2 = self.augmentor(x,  edge_index, edge_weights)
            z1 = self.main_encoder.encode_project(g1, edge_index1, edge_weights1)
            z2 = self.main_encoder.encode_project(g2, edge_index2, edge_weights2)
        else:
            g1 = self.augmentor(x) 
            g2 = self.augmentor(x) 
            z1 = self.main_encoder.encode_project(g1)
            z2 = self.main_encoder.encode_project(g2)
            
        return (z1, z2)

    def main_encode_project(self, g):
        x,  edge_index, edge_weights = g.x, g.edge_index, g.edge_weights
        return self.main_encoder.encode_project(x,  edge_index, edge_weights)


class RES(nn.Module):
    def __init__(self,):
        super(RES, self).__init__()
        self.layers = resnet18()
        
        
        self.project_head = make_project_head(1000)
        
    def forward(self, x, edge_index=None, edge_weights=None):
        return self.layers(x)
    
    def project(self, z):
        return self.project_head(z)
        
    def encode_project(self, x, edge_index=None, edge_weights=None):
        return self.project_head(self.forward(x))


# data augmented before feed in MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            make_conv(input_dim, hidden_dim),
            make_conv(hidden_dim, hidden_dim),
        )
        
        
        self.project_head =make_project_head(hidden_dim)
        
    def forward(self, x, edge_index=None, edge_weights=None):
        return self.layers(x)
    
    def project(self, z):
        return self.project_head(z)
        
    def encode_project(self, x, edge_index=None, edge_weights=None):
        return self.project_head(self.forward(x))

class Adv_sens(nn.Module):
    def __init__(self, sens_num, hidden_dim):
        super(Adv_sens, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sens_num),
        )

    def forward(self, x):
        return self.network(x)


class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)

        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        self.project_head = make_project_head(hidden_dim)

    def forward(self, x, edge_index=None, edge_weights=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weights)
            z = self.activation(z)
        return z

    def project(self, z):
        return self.project_head(z)
        
    def encode_project(self, x, edge_index=None, edge_weights=None):
        return self.project_head(self.forward(x, edge_index, edge_weights))