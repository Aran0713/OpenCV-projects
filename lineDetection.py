import cv2
import numpy as np

image = cv2.imread('./images/shapes.jpg')
cv2.imshow("First Image", image)
original = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 50, 150, apertureSize = 3)
cv2.imshow("edged", edged)


# lines = cv2.HoughLines(edged, 1, np.pi/180, 80, 5, 1)

# for line in lines: 
#     rho, theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)

#     x0 = a * rho
#     y0 = b * rho 

#     x1 = int(x0 +1000 * (-b))
#     y1 = int(y0 +1000 * (a))
#     x2 = int(x0 -1000 * (-b))
#     y2 = int(y0 -1000 * (a))
#     cv2.line(image, (x1,y1), (x2,y2), (255,0,0), 2)

# cv2.imshow("Hough Lines", image)

lines = cv2.HoughLinesP(edged, 0.5, np.pi/180, 15, 5, 20)

for x in range (0, len(lines)):
    print(lines[x])
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow("Hough Lines 2", image)

cv2.waitKey(0)
cv2.destroyAllWindows()