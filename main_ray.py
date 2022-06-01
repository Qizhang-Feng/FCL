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
import sys
from ray import tune
import argparse
#sys.path.append('./PyGCL')
import GCL.losses as L
from GCL.models import DualBranchContrast
from GCL.eval import get_split, LREvaluator


def search_model(config, checkpoint_dir=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_name = config['dataset_name']
    sens_name = config['sens_name']
    sens_num = 2 if (dataset_name=='adult' and sens_name=='gender') else 1
    TASK_TYPE = 'regression' if dataset_name=='crimes' else 'classification'

    # load dataset...
    dataset = get_dataset(dataset_name, sens_name)
    x, sens = get_samples(dataset, num=5000)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # prepare model config
    input_dim = dataset[0][0].shape[-1]
    hidden_dim = config['hidden_dim'] if dataset_name != 'celeba' else 1000
    sens_dim = dataset.sens_dim

    # create model
    mlp_main = MLP(input_dim, hidden_dim) if dataset_name != 'celeba' else RES()
    mlp_sens = MLP(sens_dim,hidden_dim)
    adv_model = Adv_sens(sens_num=sens_num, hidden_dim=hidden_dim)

    aug = FeatureDrop(drop_prob=config['drop_prob']) if dataset_name != 'celeba' else transforms.Compose([transforms.RandomCrop(size=RESIZE), transforms.ColorJitter(),
                            transforms.Grayscale(num_output_channels=3), transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])#FeatureDrop(drop_prob=config['drop_prob'])

    encoder_model = Encoder(main_encoder = mlp_main, augmentor = aug, sens_encoder = mlp_sens, adv_model=adv_model)
    encoder_model = encoder_model.to(device)

    contrast_model = DualBranchContrast(loss=L.FairInfoNCE(tau=config['tau']), mode='G2G').to(device)
    optim = Adam(encoder_model.parameters(), lr=config['lr'])
    
    # load ckpt
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state, _, = torch.load(checkpoint)
        encoder_model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)
        encoder_model = encoder_model.to(device)
    
    
    performance_list = []
    gdp_hist_list = []
    gdp_max_list = []
    gdp_kernel_list = []

    (epoch_num,interval) = (2000,40) if dataset_name!='celeba' else (10,1)


    epoch = epoch_num
    #with tqdm(total=epoch, desc='(T)') as pbar:
    for epoch in range(1, epoch+1):
        encoder_model = encoder_model.to(device)
        loss_result = train(encoder_model = encoder_model, contrast_model=contrast_model,
                                        dataloader=dataloader, optimizer = optim,
                                        conditional=config['conditional'],debias=config['debias'], adversarial=config['adversarial'] if epoch%4==0 else False,
                                        cond_temp = config['cond_temp'],
                                        debias_temp = config['debias_temp'],
                                        debias_ratio = config['debias_ratio'])
        #pbar.set_postfix({'loss': loss_result['loss'], 
        #                'conditional_loss':loss_result['conditional_loss'], 
        #                'debias_loss': loss_result['debias_loss'],
        #                'adv_loss': loss_result['adv_loss']})
        #pbar.update()

        if epoch % interval == 0:
            print(loss_result)
            result, _evaluator = test(encoder_model, dataloader, evaluator=LREvaluator(task=TASK_TYPE))
            classifier = result['classifier']
            
            # performance 
            performance = result['mae'] if dataset_name=='crimes' else result['auc']
            print('performance: ', performance)
            performance_list.append(performance)

            # fairness
            gdp_hist, gdp_kernel, gdp_max = gdp(dataset=dataset, task=TASK_TYPE, hist_num=1000, encoder_model=encoder_model, classifier=classifier, x=x, sens=sens)
            print('hist gdp: ', gdp_hist)
            gdp_hist_list.append(gdp_hist)
            print('max gdp: ', gdp_max)
            gdp_max_list.append(gdp_max)
            print('kernel gdp: ', gdp_kernel)
            gdp_kernel_list.append(gdp_kernel)


            # store ckpt
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(
                    (encoder_model.state_dict(), optim.state_dict(), {'config': config, 
                                                                    'performance_list': performance_list, 
                                                                    'hist_gdp_list': gdp_hist_list,
                                                                    'max_gdp_list': gdp_max_list,
                                                                    'kernel_gdp_list': gdp_kernel_list}), path)
            tune.report(kernel_gdp=gdp_kernel,
                        performance_dp=performance-gdp_kernel,
                        performance=performance,
                        loss=loss_result['loss'],
                        conditional_loss=loss_result['conditional_loss'], 
                        debias_loss=loss_result['debias_loss'],
                        adv_loss=loss_result['adv_loss'],
)


# %%
def main(args):

    local_dir = os.path.join('/data/qf31/ray_results', args.dataset_name, args.sens_name) if args.sens_name is not None else os.path.join('/data/qf31/ray_results', args.dataset_name)
    

    config = {
        # meta config
        'dataset_name': args.dataset_name,
        'sens_name': args.sens_name,
        'conditional': args.conditional,
        'debias': args.debias,
        'adversarial': args.adversarial,
        # tunable config
        'batch_size': tune.choice([1024*5, 1024*6, 1024*7] if args.dataset_name != 'celeba' else tune.choice([256*2, 256*3, 256*4])),
        'hidden_dim': tune.choice([200,220,240]),
        'drop_prob': tune.uniform(0.1, 0.5),
        'cond_temp': tune.uniform(1.0/400, 1.0/200),
        'debias_temp': tune.loguniform(1.0/80, 1.0/30),
        'debias_ratio': tune.uniform(2,8),
        'lr': tune.loguniform(0.00005, 0.0005),
        'tau': tune.loguniform(0.05, 0.2),
    }


    result = tune.run(
        tune.with_parameters(search_model),
        resources_per_trial={"cpu": 12, "gpu": 1},
        config=config,
        metric="performance_dp",
        mode="max",
        num_samples=args.num_samples,
        local_dir=local_dir,
    )

    return result

# %%
if __name__=='__main__':
    parser = argparse.ArgumentParser('Interface for Tabular experiments on adult/crimes')
    parser.add_argument('--dataset_name', type=str, help='dataset name', default='crimes')
    parser.add_argument('--sens_name', type=str, help='sensitive attribute name')
    parser.add_argument('--conditional', action='store_true', help='if use conditional')
    parser.add_argument('--debias', action='store_true', help='if use debias')
    parser.add_argument('--adversarial', action='store_true', help='if use adversarial')
    parser.add_argument('--num_samples', type=int, help='number of samples to tune', default=40)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    result = main(args=args)