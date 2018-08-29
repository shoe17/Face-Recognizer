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
print("Calculating bag of words")
dictionary = BOW.cluster()
print("Saving Cluster")

#np.savetxt("dictionary.py", dictionary)


#TODO make this not repetitive
for root, dirs, files in os.walk('data'):
	for name in files:
		clusterIDs = []

		image = os.path.join(root, name)
		
		if (image.lower().endswith('.jpg')):
			img = cv.imread(image)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			
			kp, dsc = sift.detectAndCompute(img, None)
		
		for descriptor in dsc:
			min = 100000000000000000000
			for cluster in range(dictionary.shape[0]):
				print(cluster)
				dist = np.linalg.norm(descriptor - dictionary[cluster])
				if (dist < min) :
					min = dist
					id = cluster

			clusterIDs.append(id)
		print(clusterIDs)
		break
				#compare each descriptor to each cluster id
				#find the min distance between the two most likely using np.linalg.norm(point-dictionary)
				#set the min dist cluster as the ID for the descriptor
				#replacing the 128 dimension vector with one number
				
"""
with open('dictionary.py', 'w') as file:
	file.write(dictionary)
"""
"""
#Convert SIFT features into cluster IDs
for root, dirs, files in os.walk('data'):
	for name in files:
		image = os.path.join(root, name):
"""
#assign clusters
