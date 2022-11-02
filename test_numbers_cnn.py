import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.datasets import mnist

import cv2
import numpy as np 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
classifier = load_model("identifying_numbers_CNN_10_Epochs.h5")

def draw_test(name, prediction, resized_img):
    BLACK = [0,0,0]
    # (src img, top thickness, bottom thickness, left thickness, right thickness, borderType, value)
    prediction_img = cv2.copyMakeBorder(resized_img, 0,0,0, resized_img.shape[0],cv2.BORDER_CONSTANT,value=(255,255,255))
    prediction_img = cv2.cvtColor(prediction_img, cv2.COLOR_GRAY2BGR)
    # print(prediction_img.shape)
    cv2.putText(prediction_img, str(prediction), (275,144), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4,(0,255,0),2)
    cv2.imshow(name, prediction_img)
    
for i in range(0,3):
    rand = np.random.randint(0,len(x_test))
    original_img = x_test[rand]
    
    # cv2.imshow("Original", original_img)
    # print(original_img.shape)

    # (src image, desired size, scale factor x, scale factor y, interpolation)
    resized_img = cv2.resize(original_img, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    # Just for 
    original_img = original_img.reshape(1,28,28,1)
    
    # cv2.imshow("Resized", resized_img)
    # print(resized_img.shape)
    
    predict_model = classifier.predict(original_img, 10, verbose=0)
    prediction = np.argmax(predict_model, axis=1)
    
    draw_test("Prediction", prediction, resized_img)
    cv2.waitKey()

cv2.destroyAllWindows()




