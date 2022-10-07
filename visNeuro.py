import cv2
import numpy as np
import random
import torch
import Pure_myKakuritsu
import torchvision_resnet_hack

class OneModel(torch.nn.Module):
        def __init__(self, base, exp):
            super(OneModel, self).__init__()
            self.base = base
            self.exp = exp
        
        def forward(self, x):
            x = self.base(x)
            x = self.exp(x)
            return x

def draw(x = [0.1, 0.2, 0.3, 0.1, 0.15], row = 2, scaleR = 50, title = 'out'):
	#tmp = np.zeros((1024,1024,3))
	# Caclulate the Resolution
	tmp = np.zeros((row*scaleR*2, row*scaleR*2, 3))
	nowX = 0
	baselineY = 0
	maxR = 0

	for i in range(row):
		for j in range(row):
			index = i * row + j
			thisR = int(x[index] * scaleR)
			nowX += thisR + 10
			cv2.circle(tmp, (nowX, baselineY + thisR + 10), thisR, (0,200,0))
			if thisR > maxR:
				maxR = thisR
			nowX += thisR

		baselineY += maxR * 2
		nowX = 0
		maxR = 0

	cv2.imwrite(title + '.bmp', tmp)
	cv2.imshow(title, tmp)
	cv2.waitKey(0)

if __name__ == "__main__":
	baseModel = torchvision_resnet_hack.resnet152(pretrained = False, ConvOnly = True)
	expModel = Pure_myKakuritsu.PureKakuritsu()
	model = OneModel(baseModel, expModel)
	checkpoint = torch.load('kakuritsu.pth', map_location = 'cpu')
	model.load_state_dict(checkpoint['state_dict'])
	print(model.exp.li1.weight[random.randint(0,1000)])
