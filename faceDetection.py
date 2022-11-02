import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')

image = cv2.imread('./images/person.jpg')
cv2.imshow("First Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
faces = face_classifier.detectMultiScale(gray, 1.3, 5)
found = 0

for (x,y,w,h) in faces:
    found = 1
    cv2.rectangle(image, (x,y), (x+w, y+h), (255, 127, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey()
    
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255, 127, 0), 2)
        cv2.imshow("image", image)
        cv2.waitKey()

        
if found == 0:
    print("No faces")


cv2.destroyAllWindows()