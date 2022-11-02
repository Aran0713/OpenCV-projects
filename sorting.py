from ast import Num
import cv2
import numpy as np



image = cv2.imread('./images/shapes.jpg')
cv2.imshow("First Image", image)
original = image.copy()

# Creating contours
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(gray, 30, 200)
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# int(len(contours))
# cv2.drawContours(image, contours, -1, (0,255,0), thickness=2)
# cv2.imshow("Contours", image)

# Printing contours
blank = np.zeros((image.shape[0], image.shape[1], 3))
cv2.drawContours(blank, contours, -1, (0,255,0), 3)
#cv2.imshow("Contours",blank)


# Sorting contours

# Sort by area
def get_area(contours):
    all_areas = []
    big_areas = []
    num = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
        
    return all_areas

# sort = sorted(contours, key=cv2.contourArea, reverse=True)
# print(get_area(sort))
#cv2.waitKey()

# i = 0
# for c in sort:
#     if (get_area(sort)[i] > 2000):
#         print(get_area(sort)[i])
#         cv2.drawContours(original, [c], -1, (255,0,0), 3)
#         cv2.imshow("Contours by area", original)
#         cv2.waitKey()
#     i = i+1


## Sort left to right
# func for sort by position
def sortPlace(contours):
    if cv2.contourArea(contours):
        M = cv2.moments(contours) # finds the moment: particular weighted avg of img pixel intensities
        return(int(M['m10'] / M['m00'])) # find the middle(x-coordinate) of the shape sorts by left to right
        # return(int(M['m01']/M['m00'])) # sorts by top to bottom 
    else:
        pass

def labelRedDot(image, c):
    M = cv2.moments(c)
    Cx = int(M['m10']/M['m00'])
    Cy = int(M['m01']/M['m00'])
    cv2.circle(image, (Cx,Cy), 5, (255,255,255), -1)
    return image
 
# for (i,c) in enumerate(contours):
#     if cv2.contourArea(c) > 2000:
#         original = labelRedDot(original, c)

# cv2.imshow("dot",original)
# cv2.waitKey()

# sortedContours = sorted(contours, key = sortPlace, reverse=False)
# area_sortedContours = get_area(sortedContours)
# num = 1
# for (i,c) in enumerate(sortedContours):
#     if (area_sortedContours[i] > 2000):
#         cv2.drawContours(image, [c], -1, (0,0,255), 3)
#         M = cv2.moments(c)
#         Cx = int(M['m10']/M['m00'])
#         Cy = int(M['m01']/M['m00'])
#         cv2.putText(image, str(num), (Cx, Cy), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 3)
#         cv2.imshow("sorted", image)
#         cv2.waitKey()
#         num =num+1
  
  
        
## ApproxPolyDP
# for c in contours:
#     cv2.drawContours(image, [c], 0, (0, 0, 255), 2)
#     cv2.imshow("Bounding rectangle", image)

# cv2.waitKey()

# for c in contours:
#     accuracy = 0.03 * cv2.arcLength(c, True)
#     approx = cv2.approxPolyDP(c, accuracy, True)
#     cv2.drawContours(original, [approx], 0, (0,255,0), 2)
#     cv2.imshow("Approx", original)
    
# cv2.waitKey()


## Convex Hull
n = len(contours) - 1
sortedContours = sorted(contours, key = sortPlace, reverse=False)[:n]

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(original, [hull], 0, (0,255,0), 2)
    cv2.imshow("Hull", original)

cv2.waitKey()

cv2.destroyAllWindows()