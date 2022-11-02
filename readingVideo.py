# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 23:50:34 2022

@author: Aran A
"""

import cv2 as cv

# use this function to reesize live videos 
def changeRes(width, height):
    capture.set(3,width)
    capture.set(4, height)


# use this function to resize pictures or videos
def rescaleFrame(frame, scale = 1.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    width = 500
    height = 500
    dimensions = (width, height)
    
    return cv.resize(frame, dimensions, interpolation = cv.INTER_AREA)
  


capture = cv.VideoCapture('Photos/dog.gif')

while True:
    isTrue, frame = capture.read()
    if (isTrue == True):
        frame_resized = rescaleFrame(frame)
        #cv.imshow('Video', frame)
        cv.imshow('Video Resized', frame_resized)
    else:
        capture = cv.VideoCapture('Photos/dog.gif')
    
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
    
capture.release()
cv.destroyAllWindows()



    
