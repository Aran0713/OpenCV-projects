# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 15:10:33 2022

@author: Aran A
"""

import cv2
import sys 

s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]
    
source = cv2.VideoCapture(s)

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: #Escape
    has_frame, frame = source.read()
    if not has_frame:
        break
        
    cv2.imshow(win_name, frame)
    
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
    
source.release()
cv2.destroyWindow(win_name)

