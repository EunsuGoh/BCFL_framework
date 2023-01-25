# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class MyData(Dataset):
#     ## Edit below here
#     def __init__(self,data,targets,*trainFlag,device_num="0"):
#       if trainFlag:
#         self.x_data=data
#         self.y_data=targets
#         self.len = len(self.x_data)
#         self.device = torch.device('cuda:'+ device_num) if torch.cuda.is_available() else torch.device('cpu')
        
#     def __getitem__(self,index):
#         if torch.cuda.is_available():
#           x_raw = self.x_data[index]
#           x_data = []

#           # 스트링 데이터를 한 글자씩 ascii로 변경 후 push
#           #data
#           for element in x_raw:
#             char_to_int = ord(element)
#             x_data.append(char_to_int)
#           x_data = torch.tensor(x_data).to(self.device)

#           #targets
#           y_data = ord(self.y_data[index])
#           y_data = torch.tensor(y_data)
#           y_data=y_data.to(self.device)
#           return x_data, y_data
    
#     def __len__(self):
#         return self.len


# class Mymodel(nn.Module):
#     def __init__(self):
#         super(Mymodel, self).__init__()
#         ## Edit Here
#         self.embed = nn.Embedding(127, 8)
#         self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
#         self.drop = nn.Dropout()
#         self.out = nn.Linear(256, 127)

#     def forward(self, x):
#         ## Edit Here
#         x = self.embed(x)
#         # print(x.shape)
#         x, hidden = self.lstm(x)
        
#         x = self.drop(x)
#         return self.out(x[:, -1, :])


# def plot_predictions(data, pred):
#     plt.scatter(
#         data[:, 0], data[:, 1], c=pred,
#         cmap='bwr')
#     plt.scatter(
#         data[:, 0], data[:, 1], c=torch.round(pred),
#         cmap='bwr', marker='+')
#     plt.show()


# if __name__ == "__main__":
#     alice_data, alice_targets, bob_data, bob_targets = XORDataset(
#         100).split_by_label()
#     plt.scatter(alice_data[:, 0], alice_data[:, 1], label='Alice')
#     plt.scatter(bob_data[:, 0], bob_data[:, 1], label='Bob')
#     plt.legend()
#     plt.show()


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

class MyData(Dataset):
    ## Edit below here
    def __init__(self,data,targets,*trainFlag,device_num="0"):
      if trainFlag:
        # _data = data.astype(np.float32) 
        _data = torch.tensor(data)
        _data = _data.type(torch.FloatTensor)
        _targets = torch.tensor(targets)
        self.x_data=_data
        self.y_data=_targets
        self.len = len(self.x_data)
        self.device = torch.device('cuda:'+ device_num) if torch.cuda.is_available() else torch.device('cpu')
        
    def __getitem__(self,index):
        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x_data[index].reshape(28,28)
          x_data =  x_data.unsqueeze(0)
          x_data = x_data.to(self.device)
          y_data = self.y_data[index].to(self.device)
          return x_data, y_data
    
    def __len__(self):
        return self.len


class Mymodel(nn.Module):
    #Edit Here
    def __init__(self): 
        super(Mymodel, self).__init__()
        # input size = 28x28 
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((W-K+2P)/S)+1 공식으로 인해 ((28-5+0)/1)+1=24 -> 24x24로 변환
        # maxpooling하면 12x12
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5) # input channel = 1, filter = 10, kernel size = 5, zero padding = 0, stribe = 1
        # ((12-5+0)/1)+1=8 -> 8x8로 변환
        # maxpooling하면 4x4
        self.drop2D = nn.Dropout2d(p=0.25, inplace=False) # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc1 = nn.Linear(320,100) # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
        self.fc2 = nn.Linear(100,62) # 100개의 출력을 10개의 출력으로 변경

    def forward(self, x):
            x = F.relu(self.mp(self.conv1(x))) # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
            x = F.relu(self.mp(self.conv2(x))) # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
            x = self.drop2D(x)
            x = x.view(x.size(0), -1) # flat
            x = self.fc1(x) # fc1 레이어에 삽입
            x = self.fc2(x) # fc2 레이어에 삽입
            return F.log_softmax(x) # fully-connected layer에 넣고 logsoftmax 적용
