import torch
import argparse
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler, DataLoader
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import tempfile
from shutil import copyfile

from sacred import Experiment
from sacred.commands import print_config
ex = Experiment('RobustDNN')
from sacred import SETTINGS 
SETTINGS['CAPTURE_MODE'] = 'no'
import pdb

# add script's parent folder to path
import os, sys
current_dir = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.dirname(current_dir)) 

# SacredDNN imports
from sacreddnn.models.robust import RobustNet, RobustDataLoader
from sacreddnn.parse_args import get_dataset, get_loss_function, get_model_type, get_optimizer
from sacreddnn.utils import num_params, l2_norm, run_and_config_to_path,\
                file_observer_dir, to_gpuid_string, take_n_per_class

def train(loss, model, device, train_loader, optimizer):
    model.train()
    t = tqdm(train_loader) # progress bar integration
    train_loss, accuracy, ndata = 0, 0, 0    
    for data, target in t:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        l = loss(output, target)
        l += model.coupling_loss()
        l.backward()
        optimizer.step()

        train_loss += l.item()*len(data)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy += pred.eq(target.view_as(pred)).sum().item()
        ndata += len(data)
        
        t.set_postfix(loss=train_loss/ndata, err=100*(1-accuracy/ndata))

def eval_loss_and_error(loss, model, device, loader):
    # t0 = time.time()
    model.eval()
    center = model.get_or_build_center().to(device)
    center.eval()

    l, accuracy = np.zeros(model.y), np.zeros(model.y)
    center_loss, center_accuracy = 0., 0.
    ensemble_loss, ensemble_accuracy = 0., 0.
    ndata = 0
    # committee_loss, committee_accuracy = 0., 0. # TODO
    with torch.no_grad():
        for data, target in loader.single_loader():

            data, target = data.to(device), target.to(device)
            
            # single replicas
            outputs = model(data, split_input=False, concatenate_output=False)
            for a, output in enumerate(outputs):
                l[a] += loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                accuracy[a] += pred.eq(target.view_as(pred)).sum().item()
            
            # ensemble 
            output = torch.mean(torch.stack(outputs), 0)
            ensemble_loss += loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            ensemble_accuracy += pred.eq(target.view_as(pred)).sum().item()

            # center
            output = center(data)
            center_loss += loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            center_accuracy += pred.eq(target.view_as(pred)).sum().item()
            ndata += len(data) 


    # print(f"@eval time: {time.time()-t0:.4f}s")
    l /= ndata
    accuracy /= ndata
    center_loss /= ndata
    center_accuracy /= ndata
    ensemble_loss /= ndata
    ensemble_accuracy /= ndata
    return l, (1-accuracy)*100, center_loss, (1-center_accuracy)*100,\
        ensemble_loss, (1-ensemble_accuracy)*100
        
@ex.config  # Configuration is defined through local variables.
def cfg():
    batch_size = 128    # input batch size for training
    epochs = 100        # number of epochs to train
    lr = 0.1            # learning rate
    weight_decay = 5e-4 # weight decay param (=L2 reg. Good value is 5e-4)
    no_cuda = False     # disables CUDA training
    nthreads = 2        # number of threads
    save_model = False  # save current model to path
    save_epoch = 10     # save every save_epoch model
    keep_models = False # keep all saved models
    load_model = ""     # load model from path
    droplr = 5          # learning rate drop factor (use 0 for no-drop)
    opt = "adam"        # optimizer type
    loss = "nll"        # classification loss [nll, mse]
    model = "lenet"     # model type  [lenet, densenet,...]  
    dataset = "cifar10" # dataset  [mnist, fashion, cifar10, cifar100]
    datapath = '~/data/'# folder containing the datasets (e.g. mnist will be in "data/MNIST")
    logtime = 2         # report every logtime epochs
    M = -1              # take only first M training examples 
    Mtest = -1          # take only first Mtest test examples 
    pclass = -1         # take only pclass training examples for each class 
    preprocess = False  # data normalization
    gpu = 0           # gpu_id to use
    dropout = 0         # dropout rate
    save_zero_epoch = False
    alpha =1            # multiplying softmax layer entries with alpha
    # ROBUST ENSEMBLE SPECIFIC
    y=1                 # number of replicas
    use_center=False    # use a central replica
    g=1e-3              # initial coupling value
    grate=1e-1          # coupling increase rate

    # Sensitivity Choice Parameters
    pooling = "max"       # pooling type
    activation_function = "relu" # activation function

    os.environ["CUDA_VISIBLE_DEVICES"] = to_gpuid_string(gpu)  # To be done before any call to torch.cuda
    


@ex.automain
def main(_run, _config):
    ## SOME BOOKKEEPING
    args = argparse.Namespace(**_config)
    print_config(_run); print()
    logdir = file_observer_dir(_run)
    if not logdir is None:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=f"{logdir}/{run_and_config_to_path(_run, _config)}")
    pdb.set_trace()
    if args.save_model: # make temp file. In the end, the model will be stored by the observers.
        save_prefix = tempfile.mkdtemp() + "/model"

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("USING CUDA =", use_cuda)
    device = torch.device(f"cuda" if use_cuda else "cpu")
    
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.nthreads)

    ## LOAD DATASET
    loader_args = {'pin_memory': True} if use_cuda else {}
    dtrain, dtest = get_dataset(args)
    if args.pclass > 0:
        train_idxs = take_n_per_class(dtrain, args.pclass)
    else:
        train_idxs = list(range(len(dtrain) if args.M <= 0 else args.M))
    
    test_idxs = list(range(len(dtest) if args.Mtest <= 0 else args.Mtest))

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
        model.load_state_dict(torch.load(args.load_model + ".pt"))
    ex.info["num_params"] = num_params(model)
    print(f"MODEL: {ex.info['num_params']} params")
    ## CREATE OPTIMIZER
    
    optimizer = get_optimizer(args, model)
    if args.droplr:
        gamma_sched = 1/args.droplr if args.droplr > 0 else 1
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,\
                milestones=[args.epochs//2, args.epochs*3//4, args.epochs*15//16], gamma=gamma_sched) 

    ## LOSS FUNCTION
    loss = get_loss_function(args)

    ## REPORT CALLBACK
    def report(epoch):
        model.eval()
 
        o = dict() # store scalar observations
        oo = dict() # store array observations
        o["epoch"] = epoch
        oo["train_loss"], oo["train_error"], o["train_center_loss"], o["train_center_error"],\
            o["train_ensemble_loss"], o["train_ensemble_error"] = \
                eval_loss_and_error(loss, model, device, train_loader)
        
        oo["test_loss"], oo["test_error"], o["test_center_loss"], o["test_center_error"],\
            o["test_ensemble_loss"], o["test_ensemble_error"] = \
                eval_loss_and_error(loss, model, device, test_loader)

        o["coupl_loss"] = model.coupling_loss().item()
        
        oo["distances"] = np.sqrt(np.array([d.item()/model.num_params() for d in model.sqdistances()]))
        oo["norms"] = np.sqrt(np.array([sqn.item()/model.num_params() for sqn in model.sqnorms()]))
        o["gamma"] = model.g
        
        print("\n", pd.DataFrame({k:[o[k]] for k in o}), "\n")
        for k in o:
            ex.log_scalar(k, o[k], epoch)
            if logdir:
                writer.add_scalar(k, o[k], epoch)
        for k in oo:
            print(f"{k}:\t{oo[k]}")
            ex.log_scalar(k, np.mean(oo[k]), epoch) # Ref. https://github.com/IDSIA/sacred/issues/465
            if logdir:
                writer.add_scalar(k, np.mean(oo[k]), epoch)
        print()

    ## START TRAINING
    report(0)
    if args.save_zero_epoch==True:
        epoch_str = '000'
        model_path = save_prefix+".pt"
        torch.save(model.state_dict(), model_path)
        if args.keep_models:
            kept_model_path = save_prefix+"_epoch_{}.pt".format(epoch_str)
            copyfile(model_path, kept_model_path)
            ex.add_artifact(kept_model_path, content_type="application/octet-stream")
        ex.add_artifact(model_path, content_type="application/octet-stream")

    for epoch in range(1, args.epochs + 1):
        epoch_str = str(epoch).zfill(3) #creates numbers like: 001, 002, ..., 010
        print(epoch_str)
        train(loss, model, device, train_loader, optimizer)
        # torch.cuda.empty_cache()
        if epoch % args.logtime == 0:
            report(epoch)
        if epoch % args.save_epoch == 0 and args.save_model:
            model_path = save_prefix+".pt"
            torch.save(model.state_dict(), model_path)
            if args.keep_models:
                kept_model_path = save_prefix+"_epoch_{}.pt".format(epoch_str)
                copyfile(model_path, kept_model_path)
                ex.add_artifact(kept_model_path, content_type="application/octet-stream")
            ex.add_artifact(model_path, content_type="application/octet-stream")
        model.increase_g()
        if args.droplr:
            scheduler.step()

    # Save model after training
    if args.save_model:
        model_path = save_prefix+"_final.pt"
        torch.save(model.state_dict(), model_path)
        ex.add_artifact(model_path, content_type="application/octet-stream")
