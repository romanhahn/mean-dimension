# generate mean, standard deviation and pca component matrix of cifar10 images
# (required for mean dimension estimate)

import torch
import argparse
from torch.utils.data import SubsetRandomSampler, DataLoader, ConcatDataset, Subset
import pandas as pd
import numpy as np
import os
import pdb
import sklearn.decomposition
from sacred import Experiment
from sacred.commands import print_config


ex = Experiment('RobustDNN')

from sacred import SETTINGS

SETTINGS['CAPTURE_MODE'] = 'no'  # don't capture output (avoid progress bar clutter)

# my imports
from sacreddnn.models.robust import RobustNet, RobustDataLoader
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer, Dataset
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path, \
    file_observer_dir, to_gpuid_string


@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128    # input batch size for training
    no_cuda = False     # disables CUDA training
    nthreads = 2        # number of threads
    model = "lenet"     # model type  [lenet, densenet,...]  
    dataset = "cifar10" # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = '~/data/'# folder containing the datasets (e.g. mnist will be in "data/MNIST")
    M = -1              # take only first M training examples 
    Mtest = -1          # take only first Mtest test examples 
    pclass = -1         # take only pclass training examples for each class 
    preprocess = False  # data normalization
    gpu = 0             # gpu_id to use
    y=1                 # number of replicates   
    os.environ["CUDA_VISIBLE_DEVICES"] = to_gpuid_string(gpu)  # To be done before any call to torch.cuda
    



@ex.automain
def main(_run, _config):
    ## SOME BOOKKEEPING
    torch.set_flush_denormal(True)
    #torch.set_default_dtype(torch.double)
    args = argparse.Namespace(**_config)
    print_config(_run);
    



    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("USING CUDA =", use_cuda)
    device = torch.device(f"cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.nthreads)

    ## LOAD DATASET

    loader_args = {'pin_memory': True} if use_cuda else {}
    DATAPATH = '~/data/'

    dtrain, dtest = get_dataset(args)

    train_idxs = list(range(len(dtrain) if args.M <= 0 else args.M))

    test_idxs = list(range(len(dtest)))

    print(f"DATASET {args.dataset}: {len(train_idxs)} Train and {len(test_idxs)} Test examples")

    train_loader = RobustDataLoader(dtrain,
                                    y=args.y, concatenate=True,
                                    sampler=SubsetRandomSampler(train_idxs),
                                    batch_size=args.batch_size, **loader_args)
    test_loader = RobustDataLoader(dtest,
                                   y=args.y, concatenate=True,
                                   sampler=SubsetRandomSampler(test_idxs),
                                   batch_size=args.batch_size, **loader_args)

    # get all data
    dataiter = iter(test_loader)

    input = torch.autograd.Variable(torch.FloatTensor()).to(device)
    output = torch.autograd.Variable(torch.LongTensor()).to(device)
    for data, target in test_loader.single_loader():
        data, target = data.to(device), target.to(device)

        input = torch.cat((input, data.detach()))
        output = torch.cat((output, target.detach()))

    


    data = input.detach()
    target = output.detach()
    #print("data shape " + str(data.shape))
    

    

    N = data.shape[0]  # number of inputs
    # transforming to 2d matrix
    xmat_all = data.view(N,-1).detach()
    xmat_all = xmat_all.cpu().numpy()
    d= xmat_all.shape[1]


    data_mean=np.mean(xmat_all, axis=0)
    data_std=np.std(xmat_all, axis=0)

    pca=sklearn.decomposition.PCA(d)
    pca.fit(xmat_all)
    comps=pca.components_

    np.save('mean.npy', data_mean)
    np.save('std.npy', data_std)
    np.save('comps.npy', comps)
    


