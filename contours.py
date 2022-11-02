import cv2
import numpy as np



image = cv2.imread('./images/shapes.jpg')
cv2.imshow("First Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow("edged", edged)

print(len(contours))
cv2.drawContours(image, contours, -1, (0,255,0), thickness=2)
cv2.imshow("edged2", image)

cv2.waitKey(0)
cv2.destroyAllWindows()