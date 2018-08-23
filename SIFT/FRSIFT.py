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

#Feature matching
"""
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 20)
search_params = dict(checks = 50)

flann = cv.FlannBasedMatcher(index_params, search_params)
sift2 = cv.xfeatures2d.SIFT_create()

bowDiction = cv.BOWImgDescriptorExtractor(sift2, cv.BFMatcher(cv.NORM_L2))
bowDiction.setVocabulary(dictionary)
print ("bow dictionary", np.shape(dictionary))
"""
#assign clusters
