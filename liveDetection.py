from time import sleep
import cv2
import numpy as np

cap = cv2.VideoCapture(1)
face_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_eye.xml')
found = 0


def face_dectector(image, size=0.5):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.05, 30)
    for (x,y,w,h) in faces:
        numOfEyes = 0
        found = 1
        cv2.rectangle(image, (x,y), (x+w, y+h), (255, 127, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        eyes = eye_classifier.detectMultiScale(roi_gray)
        sleep(.05)
        for (ex,ey,ew,eh) in eyes:
            numOfEyes = numOfEyes + 1
            if (numOfEyes > 2):
                break
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255, 127, 0), 2)

            
    image = cv2.flip(image, 1)
    return image


while True:
    ret, frame = cap.read()
    cv2.imshow("Video", face_dectector(frame))
    if cv2.waitKey(1) == 13:
        break
    
cap.release()
        
if found == 0:
    print("No faces found")


cv2.destroyAllWindows()