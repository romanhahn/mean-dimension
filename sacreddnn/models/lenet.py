import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class LeNet5(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=6, out_channels_2=16,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize1=120, fcsize2=84, nclasses=10):

        super(LeNet5, self).__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize1)
        self.fc2 = nn.Linear(fcsize1, fcsize2)
        self.fc3 = nn.Linear(fcsize2, nclasses)

    def forward(self, x):
        nsamples = x.shape[0]
        x1 = F.relu(self.conv1(x))
        x2 = self.max_pool(x1)
        x3 = F.relu(self.conv2(x2))
        x4 = self.max_pool(x3)
        x5 = x4.view(nsamples, -1)
        x6 = F.relu(self.fc1(x5))
        x7 = F.relu(self.fc2(x6))
        x8 = self.fc3(x7)
        return x8, x1, x2, x3, x4, x5, x6, x7 

class LeNet5_orig(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
            Sixth row: Choice parameters for Senstivity Analysis
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=6, out_channels_2=16,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize1=120, fcsize2=84, nclasses=10,
                 activation_function="relu", pooling="max", dropout=0):

        super(LeNet5_orig, self).__init__()
        self.activation_function = activation_function

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        if pooling=="average":
            self.pool = nn.AvgPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding)
        else:
            self.pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)

        self.drop_layer = nn.Dropout(dropout)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize1)
        self.fc2 = nn.Linear(fcsize1, fcsize2)
        self.fc3 = nn.Linear(fcsize2, nclasses)
        if self.activation_function=="tanh":
            self.activation= nn.Tanh()
        else: self.activation=nn.ReLU()



    def forward(self, x):
        nsamples = x.shape[0]
        x = F.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = x.view(nsamples, -1)
        x = self.activation(self.fc1(x))
        x = self.drop_layer(x)
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def LeNet_seq(dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10, outlier=None):
    



        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        max_pool1 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)
        max_pool2 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize)
        fc2 = nn.Linear(fcsize, nclasses)
        return torch.nn.Sequential( nn.Conv2d(in_channels, out_channels_1, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation),
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride),
            nn.ReLU(),
            nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels_2 * dim * dim, fcsize),
            nn.ReLU(),
            nn.Linear(fcsize, nclasses))


def LeNet_seq_tanh(dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10, outlier=None):
    



        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        max_pool1 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)
        max_pool2 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize)
        fc2 = nn.Linear(fcsize, nclasses)
        return torch.nn.Sequential( nn.Conv2d(in_channels, out_channels_1, kernel_size, stride),
            nn.Tanh(),
            nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation),
            nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride),
            nn.Tanh(),
            nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation),
            nn.Flatten(start_dim=1),
            nn.Linear(out_channels_2 * dim * dim, fcsize),
            nn.Tanh(),
            nn.Linear(fcsize, nclasses))

class LeNet(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10, outlier=None):

        super(LeNet, self).__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride)
        self.max_pool1 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride)
        self.max_pool2 = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize)
        self.fc2 = nn.Linear(fcsize, nclasses)

    def forward(self, x, outlier=None):
        nsamples = x.shape[0]
        x1 = F.relu(self.conv1(x))
        if outlier:
            #orig_shape=x1.shape
            #x1=x1.flatten()
            #pdb.set_trace()
            #test=torch.cuda.FloatTensor(torch.stack(nsamples*[outlier["MD_conv1"]]).shape).uniform_() > 0.5
            #x1[test]=0
            x1[torch.stack(nsamples*[outlier["MD_conv1"]])]=0
            #x1=x1.view(orig_shape)
        x2 = self.max_pool1(x1)
        if outlier:
            #orig_shape=x2.shape
            #x2=x2.flatten()
            x2[torch.stack(nsamples*[outlier["MD_max_pool1"]])]=0
            #x2=x2.view(orig_shape)
        x3 = F.relu(self.conv2(x2))
        if outlier:
            #orig_shape=x3.shape
            #x3=x3.flatten()
            x3[torch.stack(nsamples*[outlier["MD_conv2"]])]=0
            #x3=x3.view(orig_shape)
        x4 = self.max_pool2(x3)
        if outlier:
            #orig_shape=x4.shape
            #x4=x4.flatten()
            x4[torch.stack(nsamples*[outlier["MD_max_pool2"]])]=0
            #x4=x4.view(orig_shape)
        x5 = x4.view(nsamples, -1)
        x6 = F.relu(self.fc1(x5))
        if outlier:
            #orig_shape=x6.shape
            #x6=x6.flatten()
            x6[torch.stack(nsamples*[outlier["MD_fc1"]])]=0
            #x6=x6.view(orig_shape)
        x7 = self.fc2(x6)
        if outlier:
            #orig_shape=x7.shape
            #x7=x7.flatten()
            x7[torch.stack(nsamples*[outlier["MD_fc2"]])]=0
            #x7=x7.view(orig_shape)
        #x_act = [x1, x2, x3 ,x4, x5, x6]
        return x7#, x_act

class LeNet_0_bias(nn.Module):
    ''' Init Function Input:
            First row: Input parameters
            Second & Third row: Convolution parameters
            Fourth row: MaxPool parameters
            Fifth row: FC and output parameters
    '''
    def __init__(self, dim=32, in_channels=1,
                 out_channels_1=20, out_channels_2=50,
                 kernel_size=5, stride=1, padding=0, dilation=1,
                 mp_kernel_size=2, mp_stride=2, mp_padding=0, mp_dilation=1,
                 fcsize=500, nclasses=10):

        super(LeNet_0_bias, self).__init__()

        # helper for calculating dimension after conv/max_pool op
        def convdim(dim):
            return (dim + 2*padding - dilation * (kernel_size - 1) - 1)//stride + 1
        def mpdim(dim):
            return (dim + 2*mp_padding - mp_dilation * (mp_kernel_size - 1) - 1)//mp_stride + 1

        self.conv1 = nn.Conv2d(in_channels, out_channels_1, kernel_size, stride, bias=False)
        self.max_pool = nn.MaxPool2d(mp_kernel_size,
                                     stride=mp_stride,
                                     padding=mp_padding,
                                     dilation=mp_dilation)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size, stride, bias=False)

        # final dimension after applying conv->max_pool->conv->max_pool
        dim = mpdim(convdim(mpdim(convdim(dim))))
        self.fc1 = nn.Linear(out_channels_2 * dim * dim, fcsize, bias =False)
        self.fc2 = nn.Linear(fcsize, nclasses, bias=False)

    def forward(self, x):
        nsamples = x.shape[0]
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(nsamples, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
