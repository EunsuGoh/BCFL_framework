import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class XORDataset:
    """
    A dataset of uniformly scattered 2d points.
    Points with xy >= 0 are class A
    Points with xy < 0 are class B
    """

    def __init__(self, num_pts):
        self.data = 2 * torch.rand(num_pts, 2) - 1
        self.targets = (self.data[:, 0] * self.data[:, 1] < 0).float()

    def get(self):
        return self.data, self.targets

    def split(self):
        mid = len(self.data) // 2
        return self.data[:mid], self.targets[:mid], self.data[mid:], self.targets[mid:]

    def split_by_label(self):
        data_0 = self.data[self.targets == 0]
        targets_0 = self.targets[self.targets == 0]
        data_1 = self.data[self.targets == 1]
        targets_1 = self.targets[self.targets == 1]
        return data_0, targets_0, data_1, targets_1


class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return x.squeeze()


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
