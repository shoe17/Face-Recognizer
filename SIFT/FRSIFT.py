import os
import cv2 as cv
import numpy as np
from PIL import Image

sift = cv.xfeatures2d.SIFT_create()

dictionarySize = 20

BOW = cv.BOWKMeansTrainer(dictionarySize)

for root, dirs, files in os.walk('data'):
	for name in files:
		image = os.path.join(root, name)
		
		if (image.lower().endswith('.jpg')):
			img = cv.imread(image)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			
			kp, dsc = sift.detectAndCompute(img, None)
			BOW.add(dsc)

#create dictionary
dictionary = BOW.cluster()

print(dictionary)
