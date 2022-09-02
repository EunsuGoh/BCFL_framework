import h5py
import torchvision
save_root_path = "/home/dy/2cp_workspace/2CP/crowdsource_back/back/src/utils/pytorch_cifar10/data/"
model_name = "cifar-10-client-"
option="data"
# data_path = "/home/dy/2cp_workspace/2CP/crowdsource_back/back/src/utils/pytorch_cifar10/data/cifar-10-batches-py"
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=None)
print(len(trainset))

interval = len(trainset)//5

for i in range(5):
  train_file = h5py.File(save_root_path+model_name+str(i+1)+"_"+option+"_train.hdf5",'w')
  dset = train_file.create_group(model_name+str(i+1))
  dset['data'] = trainset.data[interval*i:interval*(i+1)]
  dset['targets'] = trainset.targets[interval*i:interval*(i+1)]
  # dset['preprocessed_data'] = return_dict[model_name+str(i+1)]['preprocessed_data']
  print(dset)
  train_file.close()

