import cv2
import numpy as np

image = cv2.imread('./images/space1.jpg')

# kernal blurring
# kernel3x3 = np.ones((3,3), np.float32) / 9
# kernel7x7 = np.ones((7,7), np.float32) / 9
# img = cv2.filter2D(image, -1, kernel3x3)
# img2 = cv2.filter2D(image, -1, kernel7x7)
# cv2.imshow("image2", img2)

# Other Blurring
# img = cv2.blur(image, (3,3))
# img = cv2.GaussianBlur(image, (3,3), 0)
# img = cv2.medianBlur(image, 3)
# img = cv2.bilateralFilter(image, 9, 75, 75)
# img = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)


# Sharpening
kernelSharpening = np.array([[-1,-1,-1],
                             [-1,9,-1],
                             [-1,-1,-1]])

img = cv2.filter2D(image, -1, kernelSharpening)

cv2.imshow("First Image", image)
cv2.imshow("image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()