
import numpy as np
import torch
import pdb

def main(path):

	alpha=7
	beta=0.1
	n=60000
	pi=np.pi
	split=0.8

	#x1=(2*pi)*torch.rand(n)-pi
	#x2=(2*pi)*torch.rand(n)-pi
	#x3=(2*pi)*torch.rand(n)-pi
	print(path)
	x=torch.normal(0,1,size=(n,200))
	y=torch.normal(0,1,size=(n,1))

	x_train=x[:int(n*split),:]
	y_train=y[:int(n*split),:]
	x_test=x[int(n*(1-split)):,:]
	y_test=x[int(n*(1-split)):,:]

	torch.save(x_train, path+'/x_train.pt')
	torch.save(y_train, path+'/y_train.pt')
	torch.save(x_test, path+'/x_test.pt')
	torch.save(y_test, path+'/y_test.pt')

if __name__=='__main__':
	path=sys.argv[1] # path to new folder to save data
	main(path)
