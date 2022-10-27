import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyData(Dataset):
    ## Edit below here
    def __init__(self,data,targets,*trainFlag):
      if trainFlag:
        self.x_data=data
        self.y_data=targets
        self.len = len(self.x_data)
        
    def __getitem__(self,index):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        if torch.cuda.is_available():
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


class Mymodel(nn.Module):
    def __init__(self):
        super(Mymodel, self).__init__()
        ## Edit Here
        self.embed = nn.Embedding(127, 8)
        self.lstm = nn.LSTM(8, 256, 2, batch_first=True)
        self.drop = nn.Dropout()
        self.out = nn.Linear(256, 127)

    def forward(self, x):
        ## Edit Here
        x = self.embed(x)
        x, hidden = self.lstm(x)
        x = self.drop(x)
        return self.out(x[:, -1, :])


def plot_predictions(data, pred):
    plt.scatter(
        data[:, 0], data[:, 1], c=pred,
        cmap='bwr')
    plt.scatter(
        data[:, 0], data[:, 1], c=torch.round(pred),
        cmap='bwr', marker='+')
    plt.show()


if __name__ == "__main__":
    alice_data, alice_targets, bob_data, bob_targets = XORDataset(
        100).split_by_label()
    plt.scatter(alice_data[:, 0], alice_data[:, 1], label='Alice')
    plt.scatter(bob_data[:, 0], bob_data[:, 1], label='Bob')
    plt.legend()
    plt.show()
