from email.mime import image
import cv2
import numpy as np

image = cv2.imread('./images/space3.jpg')
cv2.imshow("img1", image)
height, width = image.shape[:2]


# Translation
# shifty, shiftx = height/4, width/4
# T = np.float32([[1,0,shiftx],
#                 [0,1,shifty]])
# img_translation = cv2.warpAffine(image, T, (width,height))
# cv2.imshow("img2", img_translation)


# Rotation
# (Pivot, angle of rotation, scale)
# rotation_matrix = cv2.getRotationMatrix2D((width/2,height/2), 90, .5)
# rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
# Could also
# rotated = cv2.transpose(image)

# Flipped
flipped = cv2.flip(image, 1)



cv2.imshow("img", flipped)
cv2.waitKey(0)
cv2.destroyAllWindows()