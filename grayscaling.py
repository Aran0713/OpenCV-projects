from heapq import merge
import cv2
import numpy as np

image = cv2.imread("./images/space2.jpg")
cv2.imshow('Original', image)
#cv2.waitKey()

# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)
# cv2.waitKey()

B, G, R = cv2.split(image)
# print(B) # the value of blue in that pixel of arrays
# print(R)
# print(G)

# Value of color in image 
# zeros = np.zeros(image.shape[:2], dtype="uint8")
# cv2.imshow("Red", cv2.merge([zeros, zeros, R]))
# cv2.imshow("Green", cv2.merge([zeros, G, zeros]))
# cv2.imshow("Blue", cv2.merge([B, zeros, zeros]))

# Amount of white in grayscale is the value of the color
# cv2.imshow("Red", R) 
# cv2.imshow("Blue", B)
# cv2.imshow("Green", G)

# Amplifying the colors
# merge1 = cv2.merge([B+100, G, R])
# cv2.imshow("Merge", merge1)
# merge2 = cv2.merge([B, G+100, R])
# cv2.imshow("Merge2", merge2)
# merge3 = cv2.merge([B, G, R+100])
# cv2.imshow("Merge3", merge3)

hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv_image)
cv2.imshow('Value', hsv_image[:, :, 2])

cv2.waitKey()
cv2.destroyAllWindows()