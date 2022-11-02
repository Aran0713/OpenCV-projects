import cv2
import numpy as np

image = cv2.imread('./images/word.jpg')

kernel = np.ones((5,5), np.uint8)

# Dilation, erosion
# img = cv2.erode(image, kernel, iterations=1)
# img = cv2.dilate(image, kernel, iterations=1)
# img = cv2.morphologyEx(image,cv2.MORPH_OPEN, kernel)
# img = cv2.morphologyEx(image,cv2.MORPH_CLOSE, kernel)

# Edge detection
image = cv2.imread('./images/space1.jpg')
# img = cv2.Canny(image, 50, 120)
# img = cv2.Canny(image, 10, 170)
# img = cv2.Canny(image, 80, 100)
img = cv2.Canny(image, 60, 110)


cv2.imshow("First Image", image)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()