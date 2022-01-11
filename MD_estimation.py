### Estimates mean dimension of a neural network
# different output options (each node in the network, only last layer + softmay layer, on negative loglikelihood)


import torch
import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader, ConcatDataset, Subset
import time
import os, datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import tempfile
import pdb
import pickle
#import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.commands import print_config


ex = Experiment('RobustDNN')
# from sacred.utils import apply_backspaces_and_linefeeds # for progress bar captured output

# ex.captured_out_filter = apply_backspaces_and_linefeeds # doesn't work: sacred/issues/440
from sacred import SETTINGS

SETTINGS['CAPTURE_MODE'] = 'no'  # don't capture output (avoid progress bar clutter)

# my imports
from update_MD import update_MD
from sacreddnn.models.robust import RobustNet, RobustDataLoader
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer, Dataset
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path, \
    file_observer_dir, to_gpuid_string


@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128    # input batch size for training
    no_cuda = False     # disables CUDA training
    nthreads = 2        # number of threads
    load_model = ""     # load model from path
    loss = "nll"        # classification loss [nll, mse]
    model = "lenet"     # model type  [lenet, densenet,...]  
    dataset = "cifar10" # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = '~/data/'# folder containing the datasets (e.g. mnist will be in "data/MNIST")
    M = -1              # take only first M training examples 
    Mtest = -1          # take only first Mtest test examples 
    pclass = -1         # take only pclass training examples for each class 
    preprocess = False  # data normalization
    gpu = 0             # gpu_id to use
    pca = True
    alpha = 1
# ROBUST ENSEMBLE SPECIFIC
    y=1                 # number of replicas
    use_center=False    # use a central replica
    g=1e-3              # initial coupling value
    grate=1e-1          # coupling increase rate
    
    layer="nll"         # which node or output for mean dimension. ["nll", "last_layer_+_sm", "all_layer"]
    os.environ["CUDA_VISIBLE_DEVICES"] = to_gpuid_string(gpu)  # To be done before any call to torch.cuda
    



@ex.automain
def main(_run, _config):
    ## SOME BOOKKEEPING
    torch.set_flush_denormal(True)
    #torch.set_default_dtype(torch.double)
    args = argparse.Namespace(**_config)
    mydir = os.path.join(os.getcwd(), args.load_model[5:-3] +"_"+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(mydir)
    print_config(_run);
    

    # if saved with oberserver:
    #logdir = file_observer_dir(_run)
    #if not logdir is None:
    #    from torch.utils.tensorboard import SummaryWriter
    #    writer = SummaryWriter(log_dir=f"{logdir}/{run_and_config_to_path(_run, _config)}")

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
    ## BUILD MODEL
    Net = get_model_type(args)
    model = RobustNet(Net, y=args.y, g=args.g, grate=args.grate, use_center=args.use_center)
    model = model.to(device)
    # if use_cuda and torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model) # TODO adapt DataParallel to Robust ensemble
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location='cpu'))
    
    ex.info["num_params"] = num_params(model)
    print(f"MODEL: {ex.info['num_params']} params")

    ## LOSS FUNCTION
    loss = get_loss_function(args)

    center = model.get_or_build_center().to(device)
    center = center.eval()
    loss_entr = torch.nn.CrossEntropyLoss(reduction='none')
    sm = torch.nn.Softmax(dim=1)
    alpha =args.alpha


    model = center
    # choose outputs for which to estimate mean dimension
    agg={}
    num=1
    if args.layer=='all_layer':
        for i in model.children():
            layer_name = "layer_" + str(num)
            # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
            agg[layer_name]=((0,0,0),(0,0,0))
            num +=1
        agg['nll']=((0,0,0),(0,0,0))
        agg['sm']=((0,0,0),(0,0,0))


    if args.layer=='last_layer_+_sm':
        agg['last_layer']=((0,0,0),(0,0,0))
        agg['sm']=((0,0,0),(0,0,0))
    
    if args.layer=='nll':
        agg['nll']=((0,0,0),(0,0,0))

    

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

    # load data summary for pca transformation
    data_mean = torch.from_numpy(np.load("mean.npy")).to(device)
    data_std = torch.from_numpy(np.load("std.npy")).to(device)
    data_comp = torch.from_numpy(np.load("comps.npy")).to(device)

    # pca transformation of data
    if args.pca:
    # transforming to 2d matrix
    	xmat_all = data.view(N,-1).detach() 
    	
    	xmat_all =((xmat_all - data_mean)/data_std).detach()

    	xmat_all = torch.matmul(xmat_all, data_comp.transpose(0,1)).detach()



    k = xmat_all.shape[1]  # (images.shape[2]) * (images.shape[3])  # vector length of all features/inputs/pixel
    pca=args.pca 
    batch_size = args.batch_size
    x = xmat_all
    y = target
    # run mean dimension estimation (collecting finite differences)
    with torch.no_grad():
        for j in range(N-1): # N=number of samples
            agg= update_MD(x[j], x[j+1], y[j], y[j+1], model, pca, loss_entr, sm, args, data_mean, data_std,  data_comp, device, batch_size, agg, args.layer) 

        
        def finalize(existingAggregate):
            (count, mean, M2) = existingAggregate
            if count < 2:
                return float("nan")
            else:
                (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
                return (mean, variance, sampleVariance)

        MD={}
        i=0
        # finalize mean dimension computation by using variance of output and variance of finite differences
        for mod in agg.keys():
            
            agg[mod]=(finalize(agg[mod][0]),finalize(agg[mod][1]))
            
            MD[str(mod)]=(torch.sum(agg[mod][1][1],0)/(2*agg[mod][0][1])).cpu()
            i+=1
        
        # saving results (mean dimension and total effects (if needed))

        # save mean dimension 
        MD_file = open(mydir+"/MD_train.pkl","wb")
        pickle.dump(MD, MD_file)
        MD_file.close()
        
        
        if args.layer=='nll' & pca==False:
            # save totals 
            totals_file = open(mydir+"/totals.pkl","wb")
            pickle.dump(agg['nll'][1][1]/2,totals_file)
            totals_file.close()


