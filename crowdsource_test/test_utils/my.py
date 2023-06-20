import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

class MyData(Dataset):
    ## Edit below here
    def __init__(self,data,targets,*trainFlag,device_num="0", did_address = None, eval_flag = False):
      if trainFlag:
        # _data = data.astype(np.float32) 
        _data = torch.tensor(data)
        _data = _data.type(torch.FloatTensor)
        _targets = torch.tensor(targets)
        self.device = torch.device('cuda:'+ device_num) if torch.cuda.is_available() else torch.device('cpu')
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.02)
        ])
        self.did_address = did_address
        self.x_data=_data
        self.y_data=_targets
        self.eval_flag = eval_flag
        self.len = len(self.x_data)

    @torch.no_grad()

    def __getitem__(self,index):
        # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
          x_data = self.x_data[index].reshape(28,28)
          x_data =  x_data.unsqueeze(0)
          y_data = self.y_data[index]
          # If train client Doesn't have did, label filpped
          if self.did_address is None and self.eval_flag is False:
            x_data = self.transform(x_data)
            y_data = torch.tensor(torch.max(self.y_data)-y_data)
          x_data = x_data.to(self.device)
          y_data = y_data.to(self.device)
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
        # self.drop2D = nn.Dropout2d(p=0.25, inplace=False) # 랜덤하게 뉴런을 종료해서 학습을 방해해 학습이 학습용 데이터에 치우치는 현상을 막기 위해 사용
        self.mp = nn.MaxPool2d(2)  # 오버피팅을 방지하고, 연산에 들어가는 자원을 줄이기 위해 maxpolling
        self.fc1 = nn.Linear(320,100) # 4x4x20 vector로 flat한 것을 100개의 출력으로 변경
        self.fc2 = nn.Linear(100,62) # 100개의 출력을 10개의 출력으로 변경

    def forward(self, x):
            x = F.relu(self.mp(self.conv1(x))) # convolution layer 1번에 relu를 씌우고 maxpool, 결과값은 12x12x10
            x = F.relu(self.mp(self.conv2(x))) # convolution layer 2번에 relu를 씌우고 maxpool, 결과값은 4x4x20
            # x = self.drop2D(x)
            x = x.view(x.size(0), -1) # flat
            x = self.fc1(x) # fc1 레이어에 삽입
            x = self.fc2(x) # fc2 레이어에 삽입
            return F.log_softmax(x) # fully-connected layer에 넣고 logsoftmax 적용
    
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)  # nn.init.constant(m.bias, 0)
        self.apply(_init)
