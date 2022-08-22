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
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = myKakuritsu_Linear_Obj(9216, 128)
        # self.fc2 = nn.Linear(128, 10)
        self.fc2 = myKakuritsu_Linear_Obj(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
