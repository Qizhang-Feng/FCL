import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18



device = 'cuda' if torch.cuda.is_available() else 'cpu'

def make_conv(in_dim, out_dim):
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))


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
        
    def forward(self, x):
        x1 = self.augmentor(x)
        x2 = self.augmentor(x)
        #print(x1.shape, x2.shape)
        g1, g2 = self.main_encoder.encode_project(x1), self.main_encoder.encode_project(x2)
        return (g1, g2)


class RES(nn.Module):
    def __init__(self,):
        super(RES, self).__init__()
        self.layers = resnet18()
        
        
        self.project_head = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1000)
        )
        
    def forward(self, x):
        return self.layers(x)
    
    def project(self, z):
        return self.project_head(z)
        
    def encode_project(self, x):
        return self.project_head(self.layers(x))


# data augmented before feed in MLP
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            make_conv(input_dim, hidden_dim),
            make_conv(hidden_dim, hidden_dim),
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