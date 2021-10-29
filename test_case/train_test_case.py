# trains neural network for analytical test case

import torch
import os
import sys
import pdb
from parse_args import Dataset1
from torch.utils.data import DataLoader
import torch.optim as optim

def main(epochs, lr, act, data_path):

    data_path=data_path
    mydir=os.path.join(os.getcwd(),data_path)
    
    # inputs and outputs
    x = torch.load(data_path+"/x_train.pt")
    y = torch.load(data_path+"/y_train.pt")
    d_set=Dataset1(x,y)
    batch_size=1024
    trainloader=DataLoader(d_set,batch_size)
    #input, hidden layer, output
    D_in=x.shape[1]  
    H=300
    D_out=1
    #batch size: 
    N=20

    if act == 'ReLu':
        model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H),
                    torch.nn.ReLU(),
                    torch.nn.Linear(H, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, D_out),
        )
    if act == 'Tanh':
        model = torch.nn.Sequential(
                    torch.nn.Linear(D_in, H),
                    torch.nn.Tanh(),
                    torch.nn.Linear(H, 50),
                    torch.nn.Tanh(),
                    torch.nn.Linear(50, D_out),
        )
    # loss function
    loss_fn = torch.nn.MSELoss(reduction='sum')
    # optimizer
    optimizer=optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # learning
    learning_rate = lr
    for t in range(epochs):
        model.train()
        running_loss=0.0
        for data,target in trainloader:
            y_pred = model(data)
            loss = loss_fn(y_pred, target)
            running_loss += loss.item()
        # Zero the gradients before running the backward pass.
            model.zero_grad()
            loss.backward()
            optimizer.step()
            #with torch.no_grad():
            #    for param in model.parameters():
            #        param -= learning_rate * param.grad
        epoch_loss=running_loss/x.shape[0]
        if t % 100 == 99:
                print(t, epoch_loss)
    
    

    # model evaluation
    x = torch.load(data_path+"/x_test.pt")
    y = torch.load(data_path+"/y_test.pt")
    model.eval()
    d_set=Dataset1(x,y)
    batch_size=1024
    running_loss = 0.0
    testloader=DataLoader(d_set,batch_size)
    with torch.no_grad():
        for data, target in testloader:
            y_pred = model(data)
            loss = loss_fn(y_pred, target)
            running_loss += loss    
            test_loss = running_loss/x.shape[0]

        if not os.path.exists(data_path+'/epochs_'+str(epochs)+'_activation_'+str(act)):
                os.makedirs(data_path+'/epochs_'+str(epochs)+'_activation_'+str(act))
        torch.save(model.state_dict(), data_path+'/epochs_'+str(epochs)+'_activation_'+str(act)+"/testmodel.pt")
        torch.save(test_loss, data_path+'/epochs_'+str(epochs)+'_activation_'+str(act)+"/testloss.pt")

if __name__=="__main__":
    # epochs 
    epochs = int(sys.argv[1])
    # learning rate
    lr= float(sys.argv[2])
    # activation function
    act= sys.argv[3]
    # data_path
    data_path = sys.argv[4]

    main(epochs, lr, act, data_path)
