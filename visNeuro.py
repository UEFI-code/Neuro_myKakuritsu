import cv2
import numpy as np
import random

def draw(x = [0.1, 0.2, 0.3, 0.1, 0.15], row = 2, scaleR = 50):
	tmp = np.zeros((1024,1024,3))
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

	cv2.imwrite("out.jpg", tmp)


paint = []
for i in range(64):
	paint.append(random.random())
draw(paint, 8)
