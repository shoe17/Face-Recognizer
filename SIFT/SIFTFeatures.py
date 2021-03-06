import numpy as np
import cv2 as cv

img = cv.imread('data/face.jpg')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('data/sift_keypoints.jpg',img)

kp, des = sift.compute(gray,kp)

print("kp: ", len(kp))
print("des: ", len(des))
