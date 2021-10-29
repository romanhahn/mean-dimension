# runs full layer-by-layer analysis of analytical test case


import os
import sys
import pdb
import train_test_case
import MD_estimation
import calc_mean
import pickle
import importlib
def main(setting, random, epochs, activation, lr, train):
        #path_module = setting.replace('/','.')
        #path_module = path_module
        #print(path_module)
        setting1=os.path.basename(os.path.normpath(setting))
        module_name_full='gen_data_' + setting1
        module_name=importlib.import_module(module_name_full)
        data={} 

        for i in range(20):
                print(i)
                path_string=(setting+"/data/data_"+ str(i))
                path_final=path_string+'/epochs_'+str(epochs)+'_activation_'+str(activation)
                if not os.path.exists(path_string+'/x_train.pt'):
                    os.makedirs(path_string)
                    module_name.main(path_string)
                if random==False:
                        if not os.path.exists(path_final+'/testmodel.pt'):
                                print('random=False')
                                train_test_case.main(epochs,lr, activation, path_string)        
                else:
                        if not os.path.exists(path_final):
                                os.makedirs(path_final)
                if not os.path.exists(path_final+'/MD.pkl'):
                        MD_estimation.main(activation, path_final, random, train)
                data[str(i)]=calc_mean.main(path_final)


        data_file=open(setting+'/results_epochs_'+str(epochs)+'lr_'+str(lr)+'_activation_'+str(activation)+'.pkl','wb')
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


