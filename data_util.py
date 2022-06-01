from turtle import ycor
import numpy as np
import torch
import os
import pickle
from torch.utils.data import Dataset, DataLoader
import urllib
import os.path
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import namedtuple
from torchvision.datasets import CelebA
from PIL import Image
from torchvision import transforms

RESIZE = 128

# the dirname of this file
dir_name = os.path.dirname(os.path.abspath(__file__))

def get_samples(dataset, num=1000):
    dataloader = DataLoader(dataset, batch_size=num, shuffle=True, num_workers=4)
    x, s = None, None
    for _, data_batch in enumerate(dataloader):
        x, ys = data_batch # x.shape: B x f, y.shape: B, s.shape: B x f
        s = ys[1]
        break
    return x, s

def get_dataset(dataset_name, sens_name=None):
    assert dataset_name in ['adult', 'crimes', 'celeba']

    if dataset_name == 'adult':
        assert sens_name != None
        return Adult(sens_name=sens_name)

    if dataset_name == 'crimes':
        return Crimes()

    if dataset_name == 'celeba':
        def target_transform(attr):
            return attr[33], torch.unsqueeze(attr[20], 0) # sens and target is in {0,1}, no need to rescale
        transform = transforms.Compose([
            transforms.Resize((RESIZE,RESIZE)),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = CelebA(root=os.path.join(dir_name, 'datasets'), split='train', download=False, transform=transform, target_transform=target_transform)
        setattr(dataset, 'sens_dim', 1)
        return dataset

class Crimes(Dataset):
    def __init__(self) -> None:
        super(Crimes, self).__init__()
        x, y, s= read_crimes() 
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float() # size: B
        self.s = torch.from_numpy(s).float()
        self.s = self.s if len(self.s.shape) != 1 else torch.unsqueeze(self.s, 1)
        self.sens_dim = self.s.shape[1]
    
    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return (self.x[index], [self.y[index], self.s[index]])

class Adult(Dataset):
    def __init__(self, sens_name): 
        super(Adult, self).__init__()
        assert sens_name in ['age', 'gender']

        x, y, s = load_adult(scaler=True, sens_name=sens_name)
        self.x = torch.from_numpy(x.values).float()
        self.y = torch.from_numpy(y.values).long()
        self.s = torch.from_numpy(s.values).float() if sens_name == 'age' else torch.from_numpy(s.values)
        self.s = self.s if len(self.s.shape) != 1 else torch.unsqueeze(self.s, 1)
        self.sens_dim = self.s.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.x[index], [self.y[index], self.s[index]])



def read_dataset(name, label=None, sensitive_attribute=None, fold=None):
    if name == 'crimes':
        y_name = label if label is not None else 'ViolentCrimesPerPop'
        z_name = sensitive_attribute if sensitive_attribute is not None else 'racepctblack'
        fold_id = fold if fold is not None else 1
        return read_crimes(label=y_name, sensitive_attribute=z_name, fold=fold_id)
    if name=='adult':
        return load_adult()
    else:
        raise NotImplemented('Dataset {} does not exists'.format(name))


def read_crimes(label='ViolentCrimesPerPop', sensitive_attribute='racepctblack', fold=1, scaler=True):
    if not os.path.isfile(os.path.join(dir_name, 'datasets', 'communities.data')):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", 
            os.path.join(dir_name, 'datasets', 'communities.data'))
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            os.path.join(dir_name, 'datasets', 'communities.names'))

    # create names
    names = []
    with open(os.path.join(dir_name, 'datasets', 'communities.names'), 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(os.path.join(dir_name, 'datasets', 'communities.data'), names=names, na_values=['?'])

    to_drop = ['state', 'county', 'community', 'fold', 'communityname']
    data.fillna(0, inplace=True)
    # shuffle
    #data = data.sample(frac=1, replace=False).reset_index(drop=True)

    folds = data['fold'].astype(np.int)

    y = data[label].values
    to_drop += [label]

    s = data[sensitive_attribute].values
    to_drop += [sensitive_attribute]

    data.drop(to_drop + [label], axis=1, inplace=True)

    for n in data.columns:
        data[n] = (data[n] - data[n].mean()) / data[n].std()

    x = np.array(data.values)
    xys = np.concatenate([x, np.expand_dims(y, 1), np.expand_dims(s, 1)], axis=1)



    scaler = StandardScaler()
    xys = scaler.fit_transform(xys)
    scaler = MinMaxScaler()
    xys = scaler.fit_transform(xys) # MUST model output sigmoid and kernel gdp in range [0,1]

    x = xys[:, :-2]
    y = xys[:, -2:-1]
    s = xys[:, -1]

    return x, y, s


#This function is a minor modification from https://github.com/jmikko/fair_ERM
def load_adult(nTrain=None, scaler=True, shuffle=False, sens_name='gender'):
    if shuffle:
        print('Warning: I wont shuffle because adult has fixed test set')
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.
    Features of the Adult dataset:
    0. age: continuous.

    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    
    2. fnlwgt: continuous.

    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.

    4. education-num: continuous.

    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.

    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.

    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    
    (14. label: <=50K, >50K)
    '''
    if not os.path.isfile(os.path.join(dir_name, 'datasets', 'adult.data')):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", os.path.join(dir_name, 'datasets', 'adult.data'))
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", os.path.join(dir_name, 'datasets', 'adult.test'))
    data = pd.read_csv(
        os.path.join(dir_name, 'datasets', 'adult.data'),
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        os.path.join(dir_name, 'datasets', 'adult.test'),
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        skiprows=1, header=None
    )
    data = pd.concat([data, data_test])
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    data.reset_index(inplace=True)
    # Here we apply discretisation on column marital_status
    data.replace({' Divorced':'not married', 
                ' Married-AF-spouse':'married',
                ' Married-civ-spouse':'married',
                ' Married-spouse-absent':'married',
                ' Never-married':'not married', 
                ' Separated':'not married', 
                ' Widowed':'not married'}, inplace=True)

    data.replace({' <=50K.': '<=50K',
                    ' <=50K': '<=50K',
                    ' >50K.': '>50K',
                    ' >50K': '>50K'}, inplace=True)
    # continuous fields
    continuous_col = ['Age', 'fnlwgt', 'education-num', 'capital gain', 'capital loss', 'hours per week']

    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

    # reorder data frame and one-hot categorical fields
    data_continuous = data[continuous_col]
    data_category = data[category_col]
    data_category_onehot = pd.get_dummies(data_category, prefix=category_col)
    
    if scaler:
        scaler = StandardScaler()
        data_continuous = scaler.fit_transform(data_continuous.to_numpy())
        scaler = MinMaxScaler()
        data_continuous = scaler.fit_transform(data_continuous)
        data_continuous = pd.DataFrame(data_continuous, columns=continuous_col).reindex()

    data = pd.concat([data_continuous, data_category_onehot], axis=1, ignore_index=False)
    #return data
    # target is 2d onehot
    target = data.iloc[:,-2]
    data = data.iloc[:, :-2]

    if nTrain is None:
        nTrain = len_train
    
    #data_train = data[:nTrain]
    #data_test = data[nTrain:]
    #target_train = target[:nTrain]
    #target_test = target[nTrain:]
    # 0: 'Age'   56: 'gender_ Female' 57: 'gender_ Male'
    if sens_name == 'age':
        x = data.iloc[:, 1:]
        s = data.iloc[:, 0]
    if sens_name == 'gender':
        x = data.drop(['gender_ Female', 'gender_ Male'], axis=1) # drop 56 57
        s = data.iloc[:, 56]

    y = target

    return x, y ,s
    '''
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    #Care there is a final dot in the class only in test set which creates 4 different classes
    target = np.array([-1.0 if (val == 0 or val==1) else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    if nTrain is None:
        nTrain = len_train
    data = namedtuple('_', 'data, target')(datamat[:nTrain, :], target[:nTrain])
    data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])

    encoded_data = pd.DataFrame(data.data)
    encoded_data['Target'] = (data.target+1)/2
    to_protect = 1. * (data.data[:,9]!=data.data[:,9][0])

    encoded_data_test = pd.DataFrame(data_test.data)
    encoded_data_test['Target'] = (data_test.target+1)/2
    to_protect_test = 1. * (data_test.data[:,9]!=data_test.data[:,9][0])

    #Variable to protect (9:Sex) is removed from dataset
    return encoded_data.drop(columns=9), to_protect, encoded_data_test.drop(columns=9), to_protect_test
    '''


