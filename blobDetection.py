import cv2
import numpy as np

image = cv2.imread('./images/blubs.jpg')
cv2.imshow("First Image", image)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True;
params.minArea = 1;
params.maxArea = 10000;

detector = cv2.SimpleBlobDetector_create(params)


#detector = cv2.SimpleBlobDetector_create()
keypoints = detector.detect(image)



print(keypoints)
blank = np.zeros((1,1))
blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,0), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

cv2.imshow("Blobs", image)

cv2.waitKey(0)
cv2.destroyAllWindows()