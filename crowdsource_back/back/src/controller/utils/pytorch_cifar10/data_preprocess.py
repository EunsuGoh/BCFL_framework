import h5py
import torchvision
from dotenv import load_dotenv
import os
load_dotenv()

save_root_path = os.environ.get("CIFAR_SAVE_ROOT_PATH")
model_name = "cifar-10-client-"
option="data"

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

