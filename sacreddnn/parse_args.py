import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from .models import  lenet
#from .models import  resnet_cifar, densenet_old
from .models import  torchvision_models
from .utils import to_onehot
from .preproc_transforms import preproc_transforms

def get_xsize(args):
    xsize = [28,28]  if args.dataset in ["mnist","fashion"] else\
            [32,32]  if args.dataset in ["cifar10","cifar100"] else args.xsize
    return xsize

def get_loss_function(args):
    if args.loss == 'nll':
        def loss(output, target, **kwargs):
            
            output = F.log_softmax(output, dim=1)
            return F.nll_loss(output, target, **kwargs)
    elif args.loss == 'mse':
        def loss(output, target, **kwargs):
            target = to_onehot(target, output.shape[1])
            return F.mse_loss(output, target, **kwargs)
    # elif losstype == 'hinge': #TODO
    return loss

def get_model_type(args, outlier=None):
    dset = args.dataset
    nclasses =  10  if args.dataset in ["mnist","fashion","cifar10"] else\
                100 if args.dataset in ["cifar100"] else args.nclasses

    nchannels = 1 if args.dataset in ["mnist", "fashion"] else\
                3 if args.dataset in ["cifar10", "cifar100"] else 3

    xsize = get_xsize(args)

    

    if args.model == 'lenet' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, outlier=outlier)
    elif args.model == 'lenet_seq' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet_seq(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, outlier=outlier)
    elif args.model == 'lenet_seq_tanh' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet_seq_tanh(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, outlier=outlier)
    elif args.model == 'lenet_0_bias' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet_0_bias(dim=xsize[0], in_channels=nchannels, nclasses=nclasses)
    elif args.model == 'lenet5' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet5(dim=xsize[0], in_channels=nchannels, nclasses=nclasses)
    elif args.model == 'Lenet5_orig' and dset in ["mnist", "fashion", "cifar10", "cifar100"]:
        if xsize[0] != xsize[1]:
            raise NotImplementedError("input needs to be square for lenet")
        Net = lambda: lenet.LeNet5_orig(dim=xsize[0], in_channels=nchannels, nclasses=nclasses, activation_function=args.activation_function, pooling = args.pooling)
    elif args.model == 'densenet': #deprecated -> use densenet121
        Net = lambda: densenet_old.densenet_cifar(nclasses)
    elif args.model.startswith('wideresnet'): 
        depth = int(args.model.split('_')[1])
        widen_factor = int(args.model.split('_')[2])
        Net = lambda: wideresnet.WideResNet(depth, nclasses, widen_factor)
    elif args.model == 'preactresnet18':
        Net = lambda: preactresnet.PreActResNet18(nclasses)
    elif args.model == 'resnet110':
        Net = lambda: resnet_cifar.ResNet110(num_classes=nclasses)
    elif args.model.startswith('efficientnet'):
        Net = lambda: efficientnet.EfficientNet.from_pretrained(args.model, num_classes=nclasses)
        # Net = lambda: efficientnet.EfficientNet.from_name('efficientnet-b0', override_params={"num_classes": nclasses})
    elif args.model.startswith('mlp'):
        nin = np.prod(xsize)*nchannels
        nhs = [int(h) for h in args.model.split('_')[1:]]
        Net = lambda: mlp.MLP(nin, nhs, nclasses)
    else: # try if corresponds to one of the models import from torchvision 
        # see the file models/torchvision_models.py
        Net = lambda: eval(f"torchvision_models.{args.model}")(num_classes=nclasses)

    return Net

def get_dataset(args):
    if args.dataset == 'mnist':
        DSet = datasets.MNIST
    elif args.dataset == 'fashion':
        DSet = datasets.FashionMNIST
    elif args.dataset == 'cifar10':
        DSet = datasets.CIFAR10
    elif args.dataset == 'cifar100':
        DSet = datasets.CIFAR100
    
    if args.preprocess:
        transform_train, transform_test = preproc_transforms(args)
    else:
        transform_train = transforms.ToTensor()
        transform_test = transforms.ToTensor()
                
    dtrain = DSet(args.datapath, train=True, download=True, transform=transform_train)
    dtest = DSet(args.datapath, train=False, download=True, transform=transform_test)            
    return dtrain, dtest

def get_optimizer(args, model):
    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "nesterov":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=0.9, nesterov=True, weight_decay=args.weight_decay)
    elif args.opt == "momentum":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, 
                    momentum=0.9, nesterov=False, weight_decay=args.weight_decay)
    elif args.opt == "rmsprop": # TODO check if alpha is decay
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, 
                    alpha=0.9, momentum=0.9, weight_decay=args.weight_decay)

    return optimizer

class Dataset():
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self, index):
        return self.imgs[index,:,:,:] , self.labels[index]
        

class Dataset1():
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels
    def __len__(self):
        return self.imgs.shape[0]
    def __getitem__(self, index):
        return self.imgs[index,:] , self.labels[index]
