
# update of md estimation (using pairs of inputs)
import numpy as np
import torch
import pdb
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
sys.path.insert(0,'..')
from sacreddnn.parse_args import Dataset1
def update_MD(x, x1, label, label1, model, device, agg):
    batch_size=500
    dx = x1 - x

    #print(torch.get_default_dtype()) 

    n = x.shape[0] #number of parameters

    #################### generating helper matrix u

    u1 =torch.zeros(n,1).to(device)
    u2 =torch.ones(n,1).to(device)
    u3= torch.diag(torch.ones(n,)).to(device)
    u4 = abs(u3-1).to(device)
    u= torch.cat((u1,u2,u3,u4), dim=1)



    # generating matrix DX: colums represent vectorized (changed) images
    k = u.shape[1] #width of matrix (DX) # should be 2n+2

    ddx= (dx.repeat(k,1)).transpose(0,1).detach()


    DX= (torch.mul(ddx,u)+ (x.repeat(k,1)).transpose(0,1)).detach()
        


    # transpose to have flattened images in rows
    DX = DX.transpose(0,1).detach()
    # labels for each image
    true_label = torch.cat((label.repeat(1), label1.repeat(1), label.repeat(n), label1.repeat(n)),0).detach()
    #if pca==True:
    #    DX=(torch.mm(DX, comps.float()) + mean.float()).detach()



    #print(center)
    #preparing storing of activations
    activation = {}
    def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
    

    ## idea to "hook" all modules --- problem modules =! layers
    #for name in enter.named_children():
    #    exec("center."+name+".registerforward_hook(get_activation('"+name+"'))")
    

    #center.fc1.register_forward_hook(get_activation('fc1'))
    #center.fc2.register_forward_hook(get_activation('fc2'))
    #center.conv1.register_forward_hook(get_activation('conv1'))
    #pdb.set_trace()
    #center.max_pool1.register_forward_hook(get_activation('max_pool1'))
    #center.max_pool2.register_forward_hook(get_activation('max_pool2'))
    #center.conv2.register_forward_hook(get_activation('conv2'))
    

    #transforming data to needed form
    dx_view = DX
    #create dataset
    imgs =Dataset1(dx_view, torch.ones(10000)) 
    y = torch.zeros((k, 10)).to(device)
    #y_act = torch.zeros((k, 10)).to(device)
    #act0 = []
    act={}
    # initiate lists in dictionary
    for mod in model.children():
        act[mod]=[]
    l=1

    for mod in model.children():
        for img, lbl in DataLoader(imgs, batch_size):
            img1 = img.to(device)
            #l = img1.shape[0]
            act[mod].append(model[0:l](img1).detach())
        l+=1
        #y[i:(i+l),:] = model(img1)[0].detach()
    #pdb.set_trace()
    #del activation 
    with torch.no_grad():

            def update(existingAggregate, newValue):
                (count, mean, M2) = existingAggregate
                count += 1
                delta = newValue - mean
                mean += delta / count
                delta2 = newValue - mean
                M2 += delta * delta2
                return (count, mean, M2)
            #pdb.set_trace()
            i=0
            for mod in model.children():
                
                act[mod]=torch.cat(act[mod]).detach()

                # act: tuple; (nodes/activations in mod, finite changes for each input of nodes in mod)
                act[mod]=(act[mod][0,:],act[mod][2:(n+2),:]-act[mod][0,:] ) 
                
                # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
                agg[mod]=(update(agg[mod][0], act[mod][0]),update(agg[mod][1], act[mod][1]))
                # if any M2 summing over inputs for each node is zero
               # if 0 in agg[mod][1][2].sum(0):
               #     print('finitechanges')
               #     pdb.set_trace()
               # if 0 in agg[mod][0][2]:
               #     print('nodes')
               #     pdb.set_trace()
    return(agg )
