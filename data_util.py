import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader

class Adult(Dataset):
    def __init__(self, file_path):
        super(Adult, self).__init__()
        with open(os.path.join(file_path, 'X_train'), 'rb') as f:
            self.X_train = np.array(pickle.load(f))
        with open(os.path.join(file_path, 'y_train'), 'rb') as f:   
            self.y_train = np.array(pickle.load(f))
        
        self.male_idx = np.where(self.X_train[:, 40:41] == 0)[0]
        self.female_idx = np.where(self.X_train[:, 40:41] == 1)[0]
        self.mode = 'all'
            
    def __len__(self):
        if self.mode == 'all':
            return self.X_train.shape[0]
        if self.mode == 'male':
            return len(self.male_idx)
        if self.mode == 'female':
            return len(self.female_idx)
    def __getitem__(self, idx):
        if self.mode == 'all':
            return (torch.from_numpy(self.X_train[idx]), torch.tensor(self.y_train[idx]))
        if self.mode == 'male':
            return (torch.from_numpy(self.X_train[self.male_idx][idx]), torch.tensor(self.y_train[self.male_idx][idx]))
        if self.mode == 'female':
            return (torch.from_numpy(self.X_train[self.female_idx][idx]), torch.tensor(self.y_train[self.female_idx][idx]))
    
    def reset_data(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)