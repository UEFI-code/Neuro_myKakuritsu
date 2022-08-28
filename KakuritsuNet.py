import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from myKakuritsu_Drv import myKakuritsu_Linear_Obj as myKakuritsu_Linear_Obj

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layera = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2,stride = 2)
        )
        self.layerb = nn.Sequential(
            nn.Conv2d(16,32,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size =2,stride=2)

        )
        self.layerc = nn.Sequential(
            nn.Conv2d(32,32,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size= 2, stride = 2)
        )
        self.fc1 = myKakuritsu_Linear_Obj(1152, 1024)
        self.fc2 = myKakuritsu_Linear_Obj(1024, 1024)
        self.fc3 = myKakuritsu_Linear_Obj(1024, 1024)

    def forward(self, x):
        x = self.layera(x)
        x = self.layerb(x)
        x = self.layerc(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output
