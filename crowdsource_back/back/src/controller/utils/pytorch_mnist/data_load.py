import h5py
import torchvision
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms

import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.multiprocessing.set_start_method('spawn')
save_root_path = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_mnist/data/MNIST/MNIST"
model_name = "mnist-client-"
option="data"

class MyMnist(Dataset):
    
    def __init__(self,data,targets,*trainFlag):
      if trainFlag:
        # _data = data.astype(np.float32) 
        _data = torch.tensor(data)
        _data = _data.type(torch.FloatTensor)
        _targets = torch.tensor(targets)
        self.x_data=_data.unsqueeze(1)
        self.y_data=_targets
        self.len = len(self.x_data)
      else:
        _data = data.type(torch.FloatTensor)
        self.x_data=_data.unsqueeze(1)
        self.y_data=targets
        self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x_data[index].to(device)
          y_data = self.y_data[index].to(device)
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

train_set = torchvision.datasets.MNIST(
    root = './data/MNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor() # 데이터를 0에서 255까지 있는 값을 0에서 1사이 값으로 변환
    ])
)

# examples = enumerate(train_set)
# print(examples)
# batch_idx, (example_data, example_targets) = next(examples)
# print(example_data.shape)

train_data,train_targets = load_data(2) #dataset

# print(train_data.size())

# train_data = torch.reshape(train_data,())

my_train_data = MyMnist(train_data, train_targets, True)
print(my_train_data.x_data)

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

  