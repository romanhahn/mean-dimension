from torchvision import transforms
from .autoaugment import CIFAR10Policy

def mean_std_dataset(dataset_name):
    if dataset_name == 'cifar10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2023, 0.1994, 0.2010)
    elif dataset_name == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
    elif dataset_name == 'fashion':
        mean, std = (0.2860,), (0.3530,)
    elif dataset_name == 'imagenet': # TODO: check
        mean = (0.485, 0.456, 0.406)  
        std = (0.229, 0.224, 0.225)

    return mean, std

def preproc_transforms(args):
     ## DATA PREPROCESSING
    if args.dataset.startswith('cifar'):
        transform_train, transform_test = preproc_cifar(args)
    elif args.dataset == 'mnist' or args.dataset == 'fashion':
        transform_train, transform_test = preproc_mnist(args)
    # TODO add Imagenet

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)
    return transform_train, transform_test

def preproc_mnist(args):
    mean, std = mean_std_dataset(args.dataset)
    if args.preprocess == 1: # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    return transform_train, transform_test

def preproc_cifar(args):
    mean, std = mean_std_dataset(args.dataset)
    if args.preprocess == 1: 
        # just normalization
        transform_train = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 2: # crop, hflip, normalization
        width_crop = 32
        padding_crop = 4 

        transform_train = [
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 3: # resize, crop, hflip, normalization
        # TODO add paper reference (maybe autougment)
        size = 224
        width_crop = 224
        padding_crop = 32 

        # TODO check if normalization is correct
        transform_train = [
            transforms.Resize(size),
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

    elif args.preprocess == 4: # resize, crop, hflip, autoaugment, normalization
        # TODO add paper reference (maybe autougment)
        size = 224
        width_crop = 224
        padding_crop = 32 

        # TODO check if normalization is correct
        transform_train = [
            transforms.Resize(size),
            transforms.RandomCrop(width_crop, padding_crop),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(), # TODO: check if applies also to Cifar100
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]

        transform_test = [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    
        
    

    return transform_train, transform_test

    

