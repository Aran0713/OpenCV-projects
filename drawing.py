import cv2
import numpy as np

# Black image
image = np.zeros((512,512,3), np.uint8)


#gray scale black img
# image_bw = np.zeros((512,512), np.uint8)
# cv2.imshow("B&W", image_bw)

# line (starting point, end point, color, thickness)
# cv2.line(image, (0,0), (511, 511), (255, 127, 0), 5)


# Rectangle
# cv2.rectangle(image, (100,100), (200, 200), (255, 127, 0), -1) # -1 thickness means filled
# cv2.rectangle(image, (100,100), (200, 200), (255, 127, 0), 10)


# Circle
# cv2.circle(image, (250,250), 100, (255, 127, 0), -1)


# Polygon
# pts = np.array( [[25,25],[30,70], [70,30], [100,125]], np.int32)
# print(pts.shape)
# pts = pts.reshape((-1,1,2)) # add an extra dimension
# print(pts.shape) 
# cv2.polylines(image, [pts], True, (255, 127, 0), 3)


#Text
#(image, 'Text to Display', bottom left starting point, Font, Font Size, Color, Thickness)
String = "Hello World!"
cv2.putText(image, String, (155,290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 127, 0), 2)



cv2.imshow("Color", image)
cv2.waitKey()
cv2.destroyAllWindows()