# train
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision

# load data
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class TheModelClass(nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class CustomDataset(Dataset):
#     def __init__(self, anno_paths, transform=None):
#         super(CustomDataset, self).__init__()
#
#         self.transform = transform
#
#     def __len__(self):
#
#
#     def __getitem__(self, item):


if __name__=='__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=500,
                                              shuffle=True, num_workers=2)

    # for i, data in enumerate(trainloader):
    #     x, label = data
    #     print(x.shape)
    #     print(label.shape)
