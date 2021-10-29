### example code to reproduce experiments

# choose the neural network architecture of interest
for net in "lenet_seq" ;
#for net in "lenet_seq" "densenet121" "resnet101";
#for net in "densenet121" ;
do
	# train the model
	python robust_dnn.py -F runs_all_120/$net/single with dataset=cifar10 model=$net epochs=10 use_center=False no_cuda=False save_model=True save_epoch=10 load_model=False keep_models=True y=1 batch_size=128 gpu=0  lr=0.001 opt=adam M=-1 no_cuda=False logtime=2
	
	# run the mean dimension estimation using trained models (for different epochs of training)
	for model in runs_all_120/$net/single/1/model_epoch*.pt;
	do
		 python MD_estimation.py with dataset=cifar10 model=$net use_center=False no_cuda=False y=1 batch_size=3072 gpu=0 M=-1 load_model=$model
	done
	
done







