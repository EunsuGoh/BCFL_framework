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
from dotenv import load_dotenv
import os

load_dotenv()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
torch.multiprocessing.set_start_method('spawn')
save_root_path = os.environ.get("SHAKESPEARE_SAVE_ROOT_PATH")


class MyShakespeare(Dataset):
    
    def __init__(self,data,targets,*trainFlag):
      if trainFlag:
        # _data = data.astype(np.float32) 
        # _data = torch.tensor(data)
        # _data = _data.type(torch.FloatTensor)
        # _targets = torch.tensor(targets)
        self.x_data=data
        self.y_data=targets
        self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          # x_data = self.x_data[index].reshape(28,28)
          # x_data =  x_data.unsqueeze(0)
          x_raw = self.x_data[index]
          x_data = []
          
          # 스트링 데이터를 한 글자씩 ascii로 변경 후 push
          #data
          for element in x_raw:
            char_to_int = ord(element)
            x_data.append(char_to_int)
          x_data = torch.tensor(x_data).to(device)

          #targets
          y_data = ord(self.y_data[index])
          y_data = torch.tensor(y_data)
          y_data=y_data.to(device)
          return x_data, y_data
    
    def __len__(self):
        return self.len

##### For client data saving
trainers = ["trainer1","trainer2","trainer3","trainer4"]
evaluator = "evaluator"

train_data_path = os.environ.get("SHAKESPEARE_TRAIN_DATA_PATH")
test_data_path = os.environ.get("SHAKESPEARE_TEST_DATA_PATH")

# with open(test_data_path,'r') as f:
#   jsonfile = json.load(f)
#   users = jsonfile['users']
#   user_data = jsonfile['user_data']
#   print("users length : ", len(users))
#   print("sum of datas : ",sum(jsonfile["num_samples"]))

""" 데이터 생성이 필요할 시 주석 해제하고 사용 """
# with open("/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/"+str(evaluator)+"_data.json","w") as f:
#     json_format = {
#       "name": evaluator,
#       "x":[],
#       "y":[]
#     }
#     json.dump(json_format,f)
# for trainer in trainers:
#   with open("/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/"+str(trainer)+"_data.json","w") as f:
#     json_format = {
#       "name": trainer,
#       "x":[],
#       "y":[]
#     }
#     json.dump(json_format,f)

# # test data preprocessing
# with open(test_data_path,'r') as f:
#   data_json = json.load(f)
#   # print("train num of samples",sum(data_json["num_samples"]))
#   users = data_json['users']
#   user_data = data_json['user_data']
#   print("users length : ",len(users))
#   print("sum of test datas : ",sum(data_json["num_samples"]))
#   save_file_path = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/"
#   for idx in range(len(users)):  
#     client_data = user_data[users[idx]]
#     client_data_x = client_data["x"]
#     client_data_y = client_data["y"]
#     with open(save_file_path+"evaluator"+"_data.json",'r') as f:
#       json_object = json.load(f)
#     json_object["x"] = json_object["x"] + client_data_x
#     json_object["y"] = json_object["y"] + client_data_y
#     with open(save_file_path+"evaluator"+"_data.json",'w') as f:
#       json.dump(json_object,f)
  

# # train data preprocessing
# with open(train_data_path,'r') as f:
#   data_json = json.load(f)
#   # checklist = json.dumps(data_json)
#   # print(checklist[51300813:51300836])
#   # print("train num of samples",sum(data_json["num_samples"]))
#   users = data_json['users']
#   user_data = data_json['user_data']
#   print("users length : ",len(users))
#   print("sum of train datas : ",sum(data_json["num_samples"]))
#   save_file_path = "/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_shakespeare/data/user_data/"
#   for idx in range(len(users)):  
#     client_data = user_data[users[idx]]
#     client_data_x = client_data["x"]
#     client_data_y = client_data["y"]
#     if idx%4 ==0:
#       ## trainer1's data
#       with open(save_file_path+trainers[0]+"_data.json",'r') as f:
#         json_object = json.load(f)
#       json_object["x"] = json_object["x"] + client_data_x
#       json_object["y"] = json_object["y"] + client_data_y
#       with open(save_file_path+trainers[0]+"_data.json",'w') as f:
#         json.dump(json_object,f)
  
#     elif idx%4 == 1:
#       ## trainer2's data
#       with open(save_file_path+trainers[1]+"_data.json",'r') as f:
#         json_object = json.load(f)
#       json_object["x"] = json_object["x"] + client_data_x
#       json_object["y"] = json_object["y"] + client_data_y
#       with open(save_file_path+trainers[1]+"_data.json",'w') as f:
#         json.dump(json_object,f)
        
#     elif idx%4 ==2 :
#       ##trainer3's data
#       with open(save_file_path+trainers[2]+"_data.json",'r') as f:
#         json_object = json.load(f)
#       json_object["x"] = json_object["x"] + client_data_x
#       json_object["y"] = json_object["y"] + client_data_y
#       with open(save_file_path+trainers[2]+"_data.json",'w') as f:
#         json.dump(json_object,f)
        
#     else :
#       ## trainer4's data
#       with open(save_file_path+trainers[3]+"_data.json",'r') as f:
#         json_object = json.load(f)
#       json_object["x"] = json_object["x"] + client_data_x
#       json_object["y"] = json_object["y"] + client_data_y
#       with open(save_file_path+trainers[3]+"_data.json",'w') as f:
#         json.dump(json_object,f)
""" 데이터 생성이 필요할 시 주석 해제하고 사용 """

  # for user in users:
  #   print(user,len(data_json["user_data"][str(user)]["y"]))
    

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
#   with open("/home/dy/2cp_new/crowdsource_back/back/src/utils/pytorch_femnist/data/user_data/"+str(user)+"_data.json",'r') as f:
#     user_json = json.load(f)
#     print(len(user_json["x"]))
#     print(user_json["y"])
