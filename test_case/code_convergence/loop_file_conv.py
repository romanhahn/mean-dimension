# experimental runs to show convergence of mean dimenstion estimate
# generates data, trains model and computes mean dimension

import os
import sys
import pdb
import train_test_case
import MD_estimation_conv
import collect_MD_and_loss_conv
import pickle
import importlib
import gen_data

def main(setting, random, epochs, activation, lr, train):
        gpu=2
        os.environ["CUDA_VISIBLE_DEVICES"] ="2"
        data={} 

        for n in [50, 100, 200, 300, 400, 500, 1000, 10000, 50000, 100000]:
                print(n)                
                #generate folder with data
                path_string=(setting+"/data/data_"+ str(n))
                path_final=path_string+'/epochs_'+str(epochs)+'_activation_'+str(activation)
                if not os.path.exists(path_string+'/x_train.pt'):
                        os.makedirs(path_string)
                        # generate data (depends on n)
                        gen_data.main(path_string, n)
                if random==False:
                        # training model
                        if not os.path.exists(path_final+'/testmodel.pt'):
                                print('random=False')
                                train_test_case.main(epochs,lr, activation, path_string)        
                else:
                        if not os.path.exists(path_final):
                                os.makedirs(path_final)
                if not os.path.exists(path_final+'/MD.pkl'):
                        MD_estimation_conv.main(activation, path_final, random, train)
                data['samplesize_'+str(n)]=collect_MD_and_loss_conv.main(path_final)

        data_file=open(setting+'/results_epochs_'+str(epochs)+'_activation_'+str(activation)+'.pkl','wb')
        pickle.dump(data, data_file)
        data_file.close()       

if __name__=='__main__':

        setting=sys.argv[1] #"data/random/"
        random= sys.argv[2]=='random'
        if random:
            epochs=0
        else:
                epochs=int(sys.argv[3])
        activation=sys.argv[4]
        lr=float(sys.argv[5])
        train = sys.argv[6]=='train'
        main(setting, random, epochs, activation, lr , train)                   


