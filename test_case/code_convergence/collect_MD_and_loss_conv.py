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
	
	#load MD 
	data_path=path+'/MD_train.pkl'
	print(data_path)
	path_MD=os.path.join(os.getcwd(),data_path)
	a_file = open(path_MD, "rb")
	MD = pickle.load(a_file)
	a_file.close()
	
	#load loss
	loss=torch.load(path+"/testloss.pt")
	

	return(MD, loss)
		

   
if __name__ == "__main__":
    # path
    path = sys.argv[1]


    main(path)


