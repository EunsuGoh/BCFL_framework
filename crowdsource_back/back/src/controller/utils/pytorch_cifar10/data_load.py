import h5py
import torchvision
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.multiprocessing.set_start_method('spawn')
save_root_path = os.environ.get("CIFAR_SAVE_ROOT_PATH")
model_name = "cifar-10-client-"
option="data"

class MyCifar(Dataset):
    
    def __init__(self,data,targets):
      _data = data.astype(np.float32)
      # _data = np.ndarray()
      _data = torch.tensor(_data)
      _targets = torch.tensor(targets)
      _data = normalize(_data,p=4.0)
      _data = _data.permute((0,3,2,1))
      self.x_data=_data
      self.y_data=_targets
      self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x_data[index].to(device)
          y_data = self.y_data[index].to(device)
          return x_data, y_data
    
    def __len__(self):
        return self.len

class MyCifar_train(Subset):
    
    def __init__(self,data,targets):
      _data = data.astype(np.float32)
      # _data = np.ndarray()
      _data = torch.tensor(_data)
      _targets = torch.tensor(targets)
      _data = normalize(_data,p=4.0)
      _data = _data.permute((0,3,2,1))
      self.x_data=_data
      self.y_data=_targets
      self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x.data[index].to(device)
          y_data = self.y.data[index].to(device)
          return x_data, y_data
    
    def __len__(self):
        return self.len


def load_data(client_idx):
  data = []
  targets = []

  train_file = h5py.File(save_root_path+model_name+str(client_idx)+"_"+option+"_train.hdf5",'r')
  for key in train_file.keys():
    data = train_file[key]['data']
    targets = train_file[key]['targets']

  return data,targets


_data,_target = load_data(2)
print(_data)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
# _data = trainset.data
# _targets = trainset.targets

# mydata = MyCifar(_data,_targets)
# print(mydata.__len__())

# print(device)

# mydata.x_data.to(device)
# mydata.y_data.to(device)

  # for i in range(5):
  #   train_file = h5py.File(save_root_path+model_name+str(i+1)+"_"+option+"_train.hdf5",'r')
  #   for key in train_file.keys():
  #     data = train_file[key]['data']
  #     targets = train_file[key]['targets']
  #   print(len(data))
  #   print(len(targets))

  