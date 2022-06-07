import imp
from turtle import ycor
from GCL.augmentors.augmentor import Graph
import numpy as np
import scipy.sparse as sp
import torch
import os
from torch.utils.data import Dataset, DataLoader
import urllib
import os.path
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import namedtuple
from torchvision.datasets import CelebA
from torchvision import transforms


RESIZE = 128

# the dirname of this file
dir_name = os.path.dirname(os.path.abspath(__file__))

def normalize_(x):
    if len(x.shape) == 1: # 1-D array
        return MinMaxScaler().fit_transform(StandardScaler().fit_transform(np.expand_dims(x, axis=1)))
    else:
        return MinMaxScaler().fit_transform(StandardScaler().fit_transform(x))

def get_samples(dataset, num=1000):
    dataloader = DataLoader(dataset, batch_size=num, shuffle=True, num_workers=4)
    if dataset.__class__.__name__ in ['Pokec']:
        # graph data
        edge_index, x, ys = None, None, None

        for _, data_batch in enumerate(dataloader):
            edge_index, x, ys = data_batch # x.shape: B x f, y.shape: B, s.shape: B x f
            s = ys[1][0]
            x = x[0]
            edge_index = edge_index[0]
            break
        rand_idx = np.random.randint(0, x.shape[0], num)
        g = Graph(x=x, edge_index=edge_index, edge_weights=None)
        return g, s, rand_idx    
    else:
        x, s = None, None
        for _, data_batch in enumerate(dataloader):
            x, ys = data_batch # x.shape: B x f, y.shape: B, s.shape: B x f
            s = ys[1]
            break
        g = Graph(x=x, edge_index=None, edge_weights=None)
        return g, s

def get_dataset(dataset_name, sens_name=None):
    assert dataset_name in ['adult', 'crimes', 'celeba', 'pokecz', 'pokecn']

    if dataset_name == 'adult':
        assert sens_name != None
        assert sens_name in ['age', 'gender']
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
        setattr(dataset, 'input_dim', RESIZE)
        return dataset
    
    if dataset_name == 'pokecz':
        assert sens_name in ['age', 'region']
        sens_name = 'AGE' if sens_name=='age' else sens_name
        dataset = Pokec(dataset_name='region_job', sens_name=sens_name)
        return dataset
    
    if dataset_name == 'pokecn':
        assert sens_name in ['age', 'region']
        sens_name = 'AGE' if sens_name=='age' else sens_name
        dataset = Pokec(dataset_name='region_job_2', sens_name=sens_name)
        return dataset

'''
if args.dataset == 'pokec_z':
    dataset = 'region_job'
else:
    dataset = 'region_job_2'
sens_attr = "AGE"
sens_attr = 'region' # AGE
predict_attr = 'I_am_working_in_field'#'spoken_languages_indicator'
'''

class Pokec(Dataset):
    def __init__(self, dataset_name, sens_name, target_name='spoken_languages_indicator') -> None: 
        super(Pokec, self).__init__()
        adj, x, y, s, idx_train, idx_val, idx_test = load_pokec(dataset=dataset_name, # adj, features, labels, sens, idx_train, idx_val, idx_val
                                                    sens_attr = sens_name,
                                                    predict_attr=target_name, #  'I_am_working_in_field'(negative value in) 'spoken_languages_indicator'
                                                    path=os.path.join(dir_name, 'datasets', 'pokec'),
                                                    sens_number=500,seed=19,test_idx=False)

        self.edge_index = torch.from_numpy(np.stack([adj.tocoo().row, adj.tocoo().col], axis=0)).long()
        print('self.edge_index: ', self.edge_index.shape)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).long() # size: B
        self.s = torch.from_numpy(s).float()
        self.s = self.s if len(self.s.shape) != 1 else torch.unsqueeze(self.s, 1)
        self.sens_dim = self.s.shape[1]

        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.input_dim = self.x.shape[-1]
    
    def __len__(self):
        return 1 # only 1 large graph
        
    def __getitem__(self, index):
        return (self.edge_index, self.x, [self.y, self.s])

def load_pokec(dataset,sens_attr,predict_attr, path, sens_number=500,seed=19,test_idx=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset,path))

    idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)


    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    labels[np.where(labels > 0)] = 1 # transfer I_am_working_in_field into binary task
    

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = adj + sp.eye(adj.shape[0])

    features = (np.array(features.todense()))
    #labels = torch.LongTensor(labels)
    # adj = sparse_mx_to_torch_sparse_tensor(adj)

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0] # some label is -1 null
    random.shuffle(label_idx)

    idx_train = label_idx[:int(0.5 * len(label_idx))]
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
    if test_idx:
        idx_test = label_idx[int(0.5 * len(label_idx)):]
        idx_val = idx_test
    else:
        idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    idx_test = np.asarray(list(sens_idx & set(idx_test)))
    #sens = torch.FloatTensor(sens)
    idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
    random.seed(seed)
    random.shuffle(idx_sens_train)
    idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])

    # random.shuffle(sens_idx)
    features = normalize_(features)
    sens = normalize_(sens)
    return adj, features, labels, sens, idx_train, idx_val, idx_test
    #return adj, features, labels, idx_train, idx_val, idx_test, sens,idx_sens_train

class Crimes(Dataset):
    def __init__(self) -> None:
        super(Crimes, self).__init__()
        x, y, s= read_crimes() 
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float() # size: B
        self.s = torch.from_numpy(s).float()
        self.s = self.s if len(self.s.shape) != 1 else torch.unsqueeze(self.s, 1)
        self.sens_dim = self.s.shape[1]

        self.input_dim = self.x.shape[-1]
    
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

        self.input_dim = self.x.shape[-1]

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
    if not os.path.isfile(os.path.join(dir_name, 'datasets', 'crimes', 'communities.data')):
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data", 
            os.path.join(dir_name, 'datasets', 'crimes', 'communities.data'))
        urllib.request.urlretrieve(
            "http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names",
            os.path.join(dir_name, 'datasets', 'crimes', 'communities.names'))

    # create names
    names = []
    with open(os.path.join(dir_name, 'datasets', 'crimes', 'communities.names'), 'r') as file:
        for line in file:
            if line.startswith('@attribute'):
                names.append(line.split(' ')[1])

    # load data
    data = pd.read_csv(os.path.join(dir_name, 'datasets', 'crimes', 'communities.data'), names=names, na_values=['?'])

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
    if not os.path.isfile(os.path.join(dir_name, 'datasets', 'adult', 'adult.data')):
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", os.path.join(dir_name, 'datasets', 'adult.data'))
        urllib.request.urlretrieve(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", os.path.join(dir_name, 'datasets', 'adult.test'))
    data = pd.read_csv(
        os.path.join(dir_name, 'datasets', 'adult', 'adult.data'),
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"]
            )
    len_train = len(data.values[:, -1])
    data_test = pd.read_csv(
        os.path.join(dir_name, 'datasets', 'adult', 'adult.test'),
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


