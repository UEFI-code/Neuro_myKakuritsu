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

def draw(x = [0.1, -0.2, 0.3, -0.1, 0.15], row = 2, scaleR = 50, title = 'out'):
	#tmp = np.zeros((1024,1024,3))
	# Caclulate the Resolution
	tmp = np.zeros((row*scaleR*2, row*scaleR, 3))
	nowX = 0
	baselineY = 0
	maxR = 0

	for i in range(row):
		for j in range(row):
			index = i * row + j
			thisR = int(x[index] * scaleR)
			if thisR > 0:
				color = (0, 200, 0)
			else:
				color = (0, 0, 200)
				thisR = 0 - thisR
			nowX += thisR + 10
			cv2.circle(tmp, (nowX, baselineY + thisR + 10), thisR, color)
			nowX += thisR
			if thisR > maxR:
				maxR = thisR

		baselineY += maxR * 2
		nowX = 0
		maxR = 0

	cv2.imwrite(title + '.png', tmp)
	cv2.imshow(title, tmp)
	cv2.waitKey(0)

def testDraw():
	x = []
	for _ in range(64):
		x.append(random.random() * random.randint(-1, 1))
	draw(x, 8, 50, 'out')

if __name__ == "__main__":

	#testDraw()
	#exit(0)

	baseModel = torchvision_resnet_hack.resnet152(pretrained = False, ConvOnly = True)
	expModel = Pure_myKakuritsu.PureKakuritsu()
	model = OneModel(baseModel, expModel)
	checkpoint = torch.load('kakuritsu.pth', map_location = 'cpu')
	model.load_state_dict(checkpoint['state_dict'])
	#print(model.exp.li1.weight[random.randint(0,1000)])
	param = model.exp.li1.weight[random.randint(0,1000)].tolist()
	#draw(param, 16, 30, 'figure_kakuritsu_cell')
	print(param[0:16*16])
