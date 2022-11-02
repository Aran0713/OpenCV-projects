import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
import cv2
import numpy as np
from matplotlib import pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# print (x_train.shape)
# print(len(x_train))

# random photos from the training data set 
# for i in range(0,3):
#     rand = np.random.randint(0, len(x_train))
#     image = x_train[rand]
#     img_scaled = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
#     cv2.imshow("Data", img_scaled)
#     cv2.waitKey()  
# cv2.destroyAllWindows()

# Plot the 6 images
# plt.subplot(331)
# rand = np.random.randint(0, len(x_train))
# plt.imshow(x_train[rand], cmap=plt.get_cmap('gray'))

# plt.subplot(332)
# rand = np.random.randint(0, len(x_train))
# plt.imshow(x_train[rand], cmap=plt.get_cmap('gray'))

# plt.subplot(333)
# rand = np.random.randint(0, len(x_train))
# plt.imshow(x_train[rand], cmap=plt.get_cmap('gray'))
# plt.show()

# Prepare our dataset for training 
img_rows = x_train[0].shape[0] # 28
img_cols = x_train[0].shape[1] # 28
# reshape to add 1 at the end
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) #(60000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) # (10000, 28, 28, 1)
# shape of a single image
input_shape = (img_rows, img_cols, 1) # (28,28,1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#Normalize the data
x_train /= 255.0
x_test /= 255.0

# print(x_train.shape)
# print(x_train.shape[0])
# print(x_test.shape[0])
# print(input_shape)

##  Encode our labels (Y)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1] # y_test.shape:(10000, 10)
num_pixels = float(img_rows * img_cols) #784.0
# print(num_classes)
# print(num_pixels)
# print(y_train[0])


# Create Model #
model = Sequential()
#Convolution layer
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))  
model.add(Conv2D(64, (3,3), activation='relu'))
# Reduces the size to 12*12*64
model.add(MaxPooling2D(pool_size=(2,2)))
# Reduces overfitting
model.add(Dropout(0.25))
# Reshapes to # of elements contained in tensor
model.add(Flatten())
# Connects this layer to a fully connected/dense layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
# Output for each class
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer= SGD(0.01), metrics=['accuracy'])
# print(model.summary())

# Train Model
batch_size = 32 # larger pics use 16
epochs = 20

history = model.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data= (x_train, y_train))

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

## Plotting accuracy 
history_dict = history.history 
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(loss_values)+1)

print(epochs)
print(val_acc_values)

line1 = plt.plot(epochs, val_acc_values, label="Accuracy")
line2 = plt.plot(epochs, acc_values, label="Training Accuracy")
plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()


## Saving the model
model.save("identifying_numbers_CNN_20_Epochs.h5")
print("Model Saved")


