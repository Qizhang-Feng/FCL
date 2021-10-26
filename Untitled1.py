#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"
from data_util import *
from metric_util import *
from train_util import *
from model import *


# In[11]:


dataset_adult = Adult('./datasets/adult')
dataloader = DataLoader(dataset_adult, batch_size=2048*6, shuffle=True, num_workers=4)

dist_all = []
dist_male = []
dist_female = []
dist_male_female = []




for _ in range(50):
# In[3]:
    dataset_adult.mode = 'all'

    mlp_adult = MLP(57, 120)
    aug = FeatureDrop(drop_prob=0.2)
    encoder_model = Encoder(encoder = mlp_adult, augmentor = aug)


    # In[4]:


    contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.1), mode='G2G').to(device)
    optim = Adam(encoder_model.parameters(), lr=0.005)

    epoch = 500
    with tqdm(total=epoch, desc='(T)') as pbar:
        for epoch in range(1, epoch+1):
            loss = train(encoder_model = encoder_model, contrast_model=contrast_model, dataloader=dataloader, optimizer = optim, conditional=True)
            pbar.set_postfix({'loss': loss})
            pbar.update()


    # In[13]:


    X, y = dataset_adult[:]

    encoder_model.to('cpu')
    Z = encoder_model.encoder.encode_project(X.to('cpu'))


    # In[14]:


    dist_all.append(sng_dis(Z).detach().numpy())


    # In[15]:


    dataset_adult.mode = 'male'

    dataloader = DataLoader(dataset_adult, batch_size=2048*6, shuffle=True, num_workers=4)


    # In[16]:


    male_X, male_y = dataset_adult[:]
    male_Z = encoder_model.encoder.encode_project(male_X.to('cpu'))
    dist_male.append(sng_dis(male_Z).detach().numpy())


    # In[17]:


    dataset_adult.mode = 'female'
    dataloader = DataLoader(dataset_adult, batch_size=2048*6, shuffle=True, num_workers=4)


    # In[18]:


    female_X, male_y = dataset_adult[:]
    female_Z = encoder_model.encoder.encode_project(female_X.to('cpu'))
    dist_female.append(sng_dis(female_Z).detach().numpy())


    # In[19]:


    dist_male_female.append(dua_dis(male_Z, female_Z).detach().numpy())


# In[ ]:
    print('ave_dist_all: ', np.mean(dist_all), ' ave_dist_male: ', np.mean(dist_male), ' ave_dist_female: ', np.mean(dist_female), ' ave_dist_male_female: ', np.mean(dist_male_female))




