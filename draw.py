import cv2 as cv
import numpy as np

#img = cv.imread('Photos/cat.jpg')
#cv.imshow('Cat', img)

#creating blankscreen
blank = np.zeros((500,500,3), dtype = 'uint8')
#cv.imshow('Blank', blank)

#background
blank[:] = 0,0,255
#cv.imshow('Green', blank)

#Creating a rectangle
blank[:] = 250,250,250
cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), (0,100,100), thickness = -1) #image, rectangle start, rectangle end, color, thickness note: if you put -1 it fills it in 
#cv.imshow('rectangle', blank)

#Creating a circle
cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 40, (50,50,50), thickness = -1) # image, cneter of circle note putting it in the middle, radius, color, thickness
#cv.imshow('circle', blank)

#creating a line
cv.line(blank, (100,100), (300,400), (20,20,20), thickness = 2) #image, starting point, ending point, color, thickness
#cv.imshow('line', blank)

#Write text
cv.putText(blank, "Hello", (300, 50), cv.FONT_HERSHEY_TRIPLEX, 1.0, (255,0,0), 2) #image, text, starting point, font, font size, color, thickness
cv.imshow('Text', blank)

cv.waitKey(0)