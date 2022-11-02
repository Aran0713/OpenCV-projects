import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.datasets import mnist

import cv2
import numpy as np 

classifier = load_model("identifying_numbers_CNN_20_Epochs.h5")

original_img = cv2.imread("./images/numbers2.jpg")
gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
blurred_img = cv2.GaussianBlur(gray_img, (5,5), 0)
edged_img = cv2.Canny(blurred_img, 30, 150)
# cv2.imshow("image1", original_img)
# cv2.imshow("image2", gray_img)
# cv2.imshow("image3", blurred_img)
# cv2.imshow("image4", edged_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

# Sort by area
def get_area(contours):
    all_areas = []
    big_areas = []
    num = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
        
    return all_areas

# Sort by position
def sortPlace(contours):
    if cv2.contourArea(contours):
        M = cv2.moments(contours) # finds the moment: particular weighted avg of img pixel intensities
        return(int(M['m10'] / M['m00'])) # find the middle(x-coordinate) of the shape sorts by left to right
        # return(int(M['m01']/M['m00'])) # sorts by top to bottom 
    else:
        pass
    
 
contours, hierarchy = cv2.findContours(edged_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   
sortedContours = sorted(contours, key = sortPlace, reverse=False)
area_sortedContours = get_area(sortedContours)

numbers = []
for (i,c) in enumerate(sortedContours):
    if (area_sortedContours[i] > 30):
        M = cv2.moments(c)
        Cx = int(M['m10']/M['m00'])
        Cy = int(M['m01']/M['m00'])
        # for numbers image
        # startX = Cx - 70
        # endX = Cx + 70
        # startY = Cy - 100
        # endY = Cy + 100
        # for numbers 2 img (must be thick numbers and phone be horizontal)
        startX = Cx - 20
        endX = Cx + 20
        startY = Cy - 30
        endY = Cy + 30
        cropped_img = blurred_img[startY:endY, startX:endX]
        ret, cropped_img = cv2.threshold(cropped_img, 127, 255,cv2.THRESH_BINARY_INV)
        cropped_img = cv2.resize(cropped_img, (28,28), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("cropped", cropped_img)
        cropped_img = cropped_img.reshape(1,28,28,1)
        
        
        predict_model = classifier.predict(cropped_img, 1, verbose=0)
        prediction = str(np.argmax(predict_model, axis=1))
        numbers.append(prediction)
        
        cv2.drawContours(original_img, [c], -1, (0,0,255), 3)
        cv2.putText(original_img, prediction, (Cx, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 3)
        cv2.imshow("sorted", original_img)
        cv2.waitKey()






print(''.join(numbers))
cv2.destroyAllWindows()

