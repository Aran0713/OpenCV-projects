import cv2
import numpy as np

# print(cv2.__version__)

image = cv2.imread('./images/space1.jpg')
cv2.imshow("First Image", image)

#print(image.shape)
print("Height: ", image.shape[0])
print("Width: ", image.shape[1])
print("Depth: ", image.shape[2])

cv2.imwrite('./images/output.jpg', image)

cv2.waitKey(0)
cv2.destroyAllWindows()