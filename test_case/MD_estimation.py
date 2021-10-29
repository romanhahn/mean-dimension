# estimates mean dimension 
# for random model, create directory first and then choose it as model path

import torch
from update_MD import update_MD
import numpy as np
import os
import pdb
import sys
import pickle


def main(activation, model_path, model_random, train):
    # settings
    device=torch.device("cuda")
    print(device)
    #data_path=os.path.split(os.path.split(model_path)[0])[0]
    model_path_final=  model_path+"/testmodel.pt"

    path_to_save=  model_path
	#choose directory
    mydir = os.path.join(os.getcwd(),model_path_final)
    # load data
    if train==True:
        x= torch.load(model_path+'/../x_train.pt').to(device)
        y= torch.load(model_path+'/../y_train.pt').to(device)
    if train==False:
        x= torch.load(model_path+'/../x_test.pt').to(device)
        y= torch.load(model_path+'/../y_test.pt').to(device)
    #build model
    N=x.shape[0]
    D_in=x.shape[1]
    D_out=1
    H=300

    if activation == 'ReLu':
        model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, D_out),
        )
    if activation == 'Tanh':
        model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H),
                    torch.nn.Tanh(),
                    torch.nn.Linear(H, 50),
                    torch.nn.Tanh(),
                    torch.nn.Linear(50, D_out),
        )

    #load model
    if model_random==False:
        model.load_state_dict(torch.load(mydir, map_location="cpu"))
    model.to(device)
    model.eval()
    depth=len(model)
    agg={}
    for i in model.children():
        agg[i]=((0,0,0),(0,0,0))


    with torch.no_grad():
        for j in range(N-1): # N=number of samples
            agg= update_MD(x[j], x[j+1], y[j], y[j+1], model,device, agg)


        
        def finalize(existingAggregate):
            (count, mean, M2) = existingAggregate
            if count < 2:
                return float("nan")
            else:
                (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
                return (mean, variance, sampleVariance)

        MD={}
        i=0
        for mod in model.children():
            
            agg[mod]=(finalize(agg[mod][0]),finalize(agg[mod][1]))
            
            MD[str(mod)+str(i)]=(torch.sum(agg[mod][1][1],0)/(2*agg[mod][0][1])).cpu()
            i+=1

        if train==True: 
            MD_file = open(model_path+"/MD_train.pkl","wb")
            pickle.dump(MD, MD_file)
            MD_file.close()
        else:
            MD_file = open(model_path+"/MD_test.pkl","wb")
            pickle.dump(MD, MD_file)
            MD_file.close()
            
if __name__ == "__main__":
    # activation function
    act = sys.argv[1]
    # data_path
    model_path = sys.argv[2]
    # model (bool: from data_path or random model)
    model_random = sys.argv[3] == "random"
    # training data or test data
    train = sys.argv[4] == "train"
    
    main(act, model_path, model_random, train)
