# The Mean Dimension of Neural Networks and what it reveals


## Requirements

To install requirements:

```setup
conda env create -f environment_mean_dimension.yml
```

If necessary, datasets are downloaded in the model training procedure.

## Image Classification

### Training Models and Estimating Mean Dimension

To train a model on Cifar10, run:

```train
python robust_dnn.py -F runs_meandimension/resnet101 with model=resnet101 datapath=~/data/ epochs=120 no_cuda=False save_model=True save_epoch=10 keep_models=True batch_size=128 gpu=0 lr=0.001 logtime=2
```

The above code/bash downloads the dataset (if not already done) to "~/data/", trains the image classification task of cifar10 using the ResNet101 architecture and saves every 10th epoch while training a model.   

Available training options specific for this work are

```python
model=resnet101             # architecture: lenet_seq, lenet_seq_tanh, densenet121, resnet18, resnet34, resnet50, resnet101, resnet152
datapath=~/data/            # folder containing the datasets (e.g. cifar10 will be in "data/cifar10")
epochs=120                  # number of epochs
no_cuda=False               # disables CUDA training
save_model=True             # saves current model to path
save_epoch=10               # save model after every xth epoch of training
keep_models=True            # keep all saved models
batch_size=128              # batch size
gpu=0                       # gpu id to use
lr=0.001                    # learning rate
logtime=2                   # report every logtime epochs
```

The mean dimension estimate for the cifar10 experiments requires the data mean, data standard deviation and the pca components matrix. To generate these, run:
```prep mean dimension
python prep_MD_estimation.py
```





To estimate the mean dimension (requires mean (mean.npy), standard deviation (std.npy) and PCA components (comps.npy) of Cifar10 images in the executing directory), run:

```estimate md
python MD_estimation.py with model=resnet101 no_cuda=False batch_size=3072 gpu=0 layer=nll datapath=~/data/ load_model=runs/resnet101/model_final.pt
```
The above code/bash loads the ResNet101 architecture, loads the saved/trained model from the path in load_model and estimates the mean dimension using the negative loglikelihood as scalar output.   

Available options specific for estimating the mean dimension are

```python
model=resnet101             # architecture: lenet_seq, lenet_seq_tanh, densenet121, resnet18, resnet34, resnet50, resnet101, resnet152
no_cuda=False               # disables CUDA training
batch_size=128              # batch size
gpu=0                       # gpu id to use
layer=nll                   # node or output for mean dimension. Options: "nll", "last_layer_+_sm", "all_layer"
datapath = ~/data/          # folder containing the datasets (e.g. cifar10 will be in "data/cifar10")
load_model=runs/resnet101/model_final.pt # path to saved trained model
```

## Analytical Test Cases

### Training Models and Estimating Mean Dimension

To train the models, run this command:

```train model 
python train_test_case.py <number_of_epochs> <learning_rate> <activation_function> <path_to_data>
```

The options for activation function are: "ReLu" or "tanh". 

The data for the Ishigami experiment or the "random" experiment is generated with:
```generate Ishigami data 
python gen_data_ishigami.py <path_to_save_data>
```
or

```generate "random" data 
python gen_data_random.py <path_to_save_data>
```

To estimate the mean dimension of a given trained (or random) model and given data, run this command:
```estimate mean dimension
python MD_estimation.py <activation_function> <path_to_data> <random> <train>
```
The options for activation function are: "ReLu" or "tanh".  
'random' takes either "random" for a random model or "non_random" for a trained model.     
'train' takes either "train" for computing the mean dimension on the training data or "test" for computing it on the test dataset.




In order to get the 20 replicates and the full routine (generating data, training model and estimating mean dimesnion) run the following command for generating the ishigami data, training the model with ReLU activation and the learning rate equal to 0.01 for 80 epochs and estimate the mean dimension using the training set:
```full routine
python full_routine.py <setting> <random> <number_of_epochs> <activation_function> <learning_rate> <train>
```
In 'setting', one chooses where to generate the data, train the model (if at all) and save mean dimension estimate.  





