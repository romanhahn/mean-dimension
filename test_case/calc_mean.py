# reads pickle file (mean dimension for each node) and computes layer average mean dimension

import torch
import numpy as np
import os
import pdb
import sys
import pickle

def main(path):
	# settings
	device=torch.device("cuda")
	print(device)
	data_path=path+'/MD_train.pkl'
	print(data_path)
	path=os.path.join(os.getcwd(),data_path)

	a_file = open(path, "rb")
	MD = pickle.load(a_file)
	a_file.close()
	mean_MD = {}
	dic = {}
	MD_zeros = MD
	mean_MD_zeros ={}
	i=0
	for k in MD.keys():
		i +=1
		layer_name=str(i)+ "_" + k.split("(")[0]		
		nans= torch.isnan(MD[k])
		infs= torch.isinf(MD[k])
		nan_inf= ~nans * ~infs
		mean_MD[k]=MD[k][nan_inf].mean()
		dic[layer_name]={"mean":MD[k][nan_inf].mean(), "std": MD[k][nan_inf].std()}
		print(k, mean_MD[k])
		MD_zeros[k][~nan_inf] = 0
		mean_MD_zeros[k]= MD_zeros[k].mean()
		print('zeros')
		print(k,mean_MD_zeros[k])
	print(dic)	
	return(dic)
		

   
if __name__ == "__main__":
    # path
    path = sys.argv[1]


    main(path)


