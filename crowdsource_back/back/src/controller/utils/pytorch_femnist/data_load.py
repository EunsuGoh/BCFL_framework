from random import sample
import h5py
import torchvision
import torch
from torch.nn.functional import normalize
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
import json
import numpy as np

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.multiprocessing.set_start_method('spawn')
save_root_path = "crowdsource_back/back/src/utils/pytorch_femnist/data/user_data/"

model_name = "mnist-client-"
option="data"

class MyMnist(Dataset):
    
    def __init__(self,data,targets,*trainFlag):
      if trainFlag:
        # _data = data.astype(np.float32) 
        _data = torch.tensor(data)
        _data = _data.type(torch.FloatTensor)
        _targets = torch.tensor(targets)
        self.x_data=_data
        self.y_data=_targets
        self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x_data[index].reshape(28,28)
          x_data =  x_data.unsqueeze(0)
          x_data = x_data.to(device)
          y_data = self.y_data[index].to(device)
          return x_data, y_data
    
    def __len__(self):
        return self.len

# def load_data(client_idx):
#   data = []
#   targets = []

#   train_file = h5py.File(save_root_path+model_name+str(client_idx)+"_"+option+"_train.hdf5",'r')
#   for key in train_file.keys():
#     data = train_file[key]['data']
#     targets = train_file[key]['targets']

#   return data,targets

##### For client data saving
trainers = []
for i in range (30):
  trainers.append("trainer"+str(i+1))
print(trainers)

evaluator = "evaluator"

test_data_path = "/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/all_data_0_niid_05_keep_0_test_9.json"
train_data_path = "/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/all_data_0_niid_05_keep_0_train_9.json"



""" 데이터 생성이 필요할 시 주석 해제하고 사용 """
# /home/dy/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/user_data/evaluator_data.json
with open("/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/user_data/"+str(evaluator)+"_data.json","w") as f:
    json_format = {
      "name": evaluator,
      "x":[],
      "y":[]
    }
    json.dump(json_format,f)
for trainer in trainers:
  with open("/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/user_data/"+str(trainer)+"_data.json","w") as f:
    json_format = {
      "name": trainer,
      "x":[],
      "y":[]
    }
    json.dump(json_format,f)

# test data preprocessing
with open(test_data_path,'r') as f:
  data_json = json.load(f)
  # print("train num of samples",sum(data_json["num_samples"]))
  users = data_json['users']
  user_data = data_json['user_data']
  print("users length : ",len(users))
  print("sum of test datas : ",sum(data_json["num_samples"]))
  save_file_path = "/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/user_data/"
  for idx in range(len(users)):  
    client_data = user_data[users[idx]]
    client_data_x = client_data["x"]
    client_data_y = client_data["y"]
    with open(save_file_path+"evaluator"+"_data.json",'r') as f:
      json_object = json.load(f)
    json_object["x"] = json_object["x"] + client_data_x
    json_object["y"] = json_object["y"] + client_data_y
    with open(save_file_path+"evaluator"+"_data.json",'w') as f:
      json.dump(json_object,f)
  

# train data preprocessing
with open(train_data_path,'r') as f:
  data_json = json.load(f)
  # print("train num of samples",sum(data_json["num_samples"]))
  users = data_json['users']
  user_data = data_json['user_data']
  print("users length : ",len(users))
  print("sum of train datas : ",sum(data_json["num_samples"]))
  save_file_path = "/media/hdd1/es_workspace/BCFL_framework_es/crowdsource_back/back/src/controller/utils/pytorch_femnist/data/user_data/"
  trainer_idx = 0
  for idx in range(len(users)): 
    # print(idx) 
    client_data = user_data[users[idx]]
    client_data_x = client_data["x"]
    client_data_y = client_data["y"]
    if idx%30 ==0:
      # ## start from trainer1
      trainer_idx = 0
      with open(save_file_path+trainers[trainer_idx]+"_data.json",'r') as f:
        json_object = json.load(f)
      json_object["x"] = json_object["x"] + client_data_x
      json_object["y"] = json_object["y"] + client_data_y
      with open(save_file_path+trainers[trainer_idx]+"_data.json",'w') as f:
        json.dump(json_object,f)
      trainer_idx+=1
  
    else:
      with open(save_file_path+trainers[trainer_idx]+"_data.json",'r') as f:
        json_object = json.load(f)
      json_object["x"] = json_object["x"] + client_data_x
      json_object["y"] = json_object["y"] + client_data_y
      with open(save_file_path+trainers[trainer_idx]+"_data.json",'w') as f:
        json.dump(json_object,f)
      trainer_idx+=1
    # elif idx%4 ==2 :
    #   ##trainer3's data
    #   with open(save_file_path+trainers[2]+"_data.json",'r') as f:
    #     json_object = json.load(f)
    #   json_object["x"] = json_object["x"] + client_data_x
    #   json_object["y"] = json_object["y"] + client_data_y
    #   with open(save_file_path+trainers[2]+"_data.json",'w') as f:
    #     json.dump(json_object,f)
        
    # else :
    #   ## trainer4's data
    #   with open(save_file_path+trainers[3]+"_data.json",'r') as f:
    #     json_object = json.load(f)
    #   json_object["x"] = json_object["x"] + client_data_x
    #   json_object["y"] = json_object["y"] + client_data_y
    #   with open(save_file_path+trainers[3]+"_data.json",'w') as f:
    #     json.dump(json_object,f)
""" 데이터 생성이 필요할 시 주석 해제하고 사용 """

#   for user in users:
#     print(user,len(data_json["user_data"][str(user)]["y"]))
    

# with open(test_data_path,'r') as f:
#   data_json = json.load(f)
#   print("test num of samples",sum(data_json["num_samples"]))





# users = data_json['users']
# samples_per_user = data_json['num_samples']
# data = data_json['user_data']

# for user in user_in_use :
#   with open(save_root_path+str(user)+"_data.json", 'w') as f:
#     json_object = data[str(user)]
#     print(json_object)
#     json.dump(json_object,f)

# for user in users:
#   with open("crowdsource_back/back/src/utils/pytorch_femnist/data/user_data/"+str(user)+"_data.json",'r') as f:
#     user_json = json.load(f)
#     print(len(user_json["x"]))
#     print(user_json["y"])
