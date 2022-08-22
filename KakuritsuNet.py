import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import myKakuritsu_Linear

class myKakuritsu_Linear_Function(torch.autograd.Function):
    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, input, weight, Kakuritsu):
        ctx.save_for_backward(input, weight)
        #output = input.mm(weight.t())
        #output = mylinear_cpp.forward(input, weight)
        output = myKakuritsu_Linear.forward(input, weight, Kakuritsu)

        return output[0]

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        #print(grad_output)
        #grad_input = grad_weight = None
        #grad_input = grad_output.mm(weight)
        #grad_weight = grad_output.t().mm(input)
        #grad_input, grad_weight = mylinear_cpp.backward(grad_output, input, weight)
        grad_input, grad_weight = myKakuritsu_Linear.backward(grad_output, input, weight)

        #print(grad_input)

        return grad_input, grad_weight, grad_weight

class myKakuritsu_Linear_Obj(nn.Module):
    def __init__(self, input_features, output_features):
        super(myKakuritsu_Linear_Obj, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight.data.uniform_(-0.1, 0.1)
        self.Kakuritsu = nn.Parameter(torch.ones(output_features, input_features) * 0.8)

    def forward(self, input):
        return myKakuritsu_Linear_Function.apply(input, self.weight, self.Kakuritsu)

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
