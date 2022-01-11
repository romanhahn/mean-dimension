# function that computes finite differences for each feature for 2 inputis (x and x1)
import numpy as np
import torch
import pdb
from sacreddnn.parse_args import Dataset
from torch.utils.data import DataLoader #, Subset


def update_MD(x, x1, label, label1, center, pca, loss_entr, sm, args, data_mean, data_std, data_comp, device, batch_size, agg, layer):

    model=center
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



    #PCA-inverse transformation
    if pca==True:
        DX= torch.mm(DX, torch.inverse(data_comp.transpose(0,1))).detach()
        DX= (torch.mul(DX, data_std) + data_mean).detach()

    #transforming data to required form
    dx_view = DX.view((-1, 3, 32, 32))
    #create dataset
    imgs =Dataset(dx_view, torch.ones(10000)) 
    act={}
    


    for mod in agg.keys():
        act[mod]=[]
   
    # choosing proper output choice
    if layer=='all_layer':
        l=1
        for mod in agg.keys():
            for img, lbl in DataLoader(imgs, batch_size):
                img1 = img.to(device)
                # act is a dict where each key has a value that contains a list of node activations (dimension depends on layer/mod)
                act[mod].append(model[0:l](img1).detach())
            l+=1
    if layer=='nll':
        for img, lbl in DataLoader(imgs, batch_size):
            img1 = img.to(device)
            # act is a dict where each key has a value that contains a list of node activations (dimension depends on layer/mod)
            act['nll'].append(model(img1).detach())

    if layer=='last_layer_+_sm':
        for img, lbl in DataLoader(imgs, batch_size):
            img1 = img.to(device)
            # act is a dict where each key has a value that contains a list of node activations (dimension depends on layer/mod)
            act['last_layer'].append(model(img1).detach())

    #with torch.no_grad():

    
    #updating mean and variance of finite changes (thus we dont need to save all data; only these aggregates)
    def update(existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)


    
    if layer=='nll':

        # getting finite changes from nll
        act_temp=torch.cat(act['nll']).detach()
        nll_temp=loss_entr(act_temp, true_label)
        finite_changes = nll_temp[2:(n+2)]-nll_temp[0]
        agg['nll']=(update(agg['nll'][0], nll_temp[0]),update(agg['nll'][1], finite_changes))
    



    if layer=='last_layer_+_sm':
    

        # getting finite changes from last layer scores   
        act_temp=torch.cat(act['last_layer']).detach()
        
        act['last_layer']=(act_temp[0,:],act_temp[2:(n+2),:]-act_temp[0,:] ) 
        
        # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
        agg['last_layer']=(update(agg['last_layer'][0], act['last_layer'][0]),update(agg['last_layer'][1], act['last_layer'][1]))
        
        # getting finite changes from class probabilities   
        sm_temp=torch.nn.functional.softmax(act_temp,dim=1)
        act['sm']=(sm_temp[0,:],sm_temp[2:(n+2),:]-sm_temp[0,:] ) 
        
        # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
        agg['sm']=(update(agg['sm'][0], act['sm'][0]),update(agg['sm'][1], act['sm'][1]))
        

        


    i=0
    if layer=='all_layer':
        # only for for modules in model (without softmax and nll):
        modules_list= list(agg.keys())[:-2]
        for mod in modules_list:
            
            # cat of the list (values of dict are lists)
            act[mod]=torch.cat(act[mod]).detach()
            # act: tuple of dim 2; (nodes/activations in layer/mod, finite changes for each input of nodes in mod)
            act[mod]=(act[mod][0,:],act[mod][2:(n+2),:]-act[mod][0,:] ) 
            
            # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
            agg[mod]=(update(agg[mod][0], act[mod][0]),update(agg[mod][1], act[mod][1]))

        
        # getting finite changes from nll
        act_temp=torch.cat(act['nll']).detach()
        nll_temp=loss_entr(act_temp, true_label)
        finite_changes = nll_temp[2:(n+2)]-nll_temp[0]
        agg['nll']=(update(agg['nll'][0], nll_temp[0]),update(agg['nll'][1], finite_changes))

        # getting finite changes from class probabilities   
        sm_temp=torch.nn.functional.softmax(act_temp,dim=1)
        act['sm']=(act_temp[0,:],act_temp[2:(n+2),:]-act_temp[0,:] ) 
        
        # agg: tuple; (tuple of (count, mean, M2) of activations/nodes in mod , tuple of (count, mean, M2) of finite changes for each input of nodes in mod)
        agg['sm']=(update(agg['sm'][0], act['sm'][0]),update(agg['sm'][1], act['sm'][1]))


    return(agg)
    
