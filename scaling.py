import cv2
import numpy as np

image = cv2.imread('./images/space4.jpg')


# Linear
# scaled = cv2.resize(image, None, fx=2, fy=2)
#Other scaling
# scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
# scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
# scaled = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)

# Cropping
height, width = image.shape[:2]
startx, starty = int(width*.10), int(height*.10)
endx, endy = int(width*0.9), int(height*0.9)
img = image[startx:endy, starty:endx]
cv2.imshow("img", img)

# rectangle
cv2.rectangle(image, (startx, starty), (endx, endy), (0,255,255), 10)

cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
