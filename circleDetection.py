import cv2
import numpy as np

image = cv2.imread('./images/shapes.jpg')
cv2.imshow("First Image", image)
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 5)
cv2.imshow("blur", blur)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 20)
circles = np.uint16(np.around(circles))

print(circles)
print(circles[0,:])
for i in circles[0,:]:
    cv2.circle(image, (i[0], i[1]), i[2], (0,255,0), 2)
    cv2.circle(image, (i[0],i[1]), 2, (0,255,0), 5)


cv2.imshow("Circles", image)

cv2.waitKey(0)
cv2.destroyAllWindows()