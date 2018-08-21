import numpy as np
import cv2
from matplotlib import pyplot as plt
img1 = cv2.imread('data/face.jpg')          # queryImage
img2 = cv2.imread('data/work_face.jpg') # trainImage

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

#Flann parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

matchesMask = [[0, 0] for i in range(len(matches))]


# Apply ratio test
for i, (m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0, 255, 0),
					singlePointColor = (255, 0, 0),
					matchesMask = matchesMask,
					flags = 0)
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches, None, **draw_params)
plt.imshow(img3),plt.show()
