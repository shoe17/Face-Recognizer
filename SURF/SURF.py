import numpy as np
import cv2 as cv

img = cv.imread('data/Butterfly.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

surf = cv.xfeatures2d.SURF_create(400)
surf.setHessianThreshold(50000)

kp, des = surf.detectAndCompute(gray,None)

img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imwrite('data/surf_keypoints.jpg',img)


