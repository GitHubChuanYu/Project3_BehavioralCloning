#Import necessary libraries for training data reading and processing
import csv
import cv2
import numpy as np
import sklearn
from random import shuffle

#Read training data into lines including camera images and steering angle
lines = []
with open('Train_Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

#Split data between train dataset and validation dataset
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#Define generator for processing large amounts of data memory-efficiently
def generator(samples, batch_size=32):
    steering_correction_left = 0.1 #steering angle correction for left camera image
    steering_correction_right = 0.1 #steering angle correction for right camera image
    num_samples = len(samples)
    while 1: #Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            augmented_images, augmented_measurements = [], []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split("\\")[-1]
                    current_path = 'Train_Data/IMG/' + filename #Create correct current image path
                    image = cv2.cvtColor(cv2.imread(current_path),cv2.COLOR_BGR2RGB) #Convert BGR training image to RGB type
                    if i==0:
                        #For center camera image, no steering angle correction
                        measurement = float(batch_sample[3]) 
                    elif i==1:
                        #For left camera image, reduce steering angle for left turn(negative value), and increase steering angle for right turn(positive value)
                        measurement = float(batch_sample[3]) + steering_correction_left 
                    elif i==2:
                        #For right camera image, increase steering angle for left turn(negative value), and reduce steering angle for right turn(positive value)
                        measurement = float(batch_sample[3]) - steering_correction_right
                    images.append(image)
                    measurements.append(measurement)
            #Data Augmentation --> flipping images and steering measurements
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image,1))
                augmented_measurements.append(measurement*-1.0)
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            #Output piece of data from generator
            yield sklearn.utils.shuffle(X_train,y_train)    

#Create two generators for training and validation datasets
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

#Import necessary tensorflow tools from keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

#Training neural network using keras
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))#Data normalization
model.add(Cropping2D(cropping=((70,25),(0,0))))#Cropping images in keras
#Nivida autonomous vehicle training network
#The Nvidia network starts with five convolutional layers
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
#Followed by four fully connected layers
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10)) 
model.add(Dense(1))

#Neural network training using "mse" as loss function and adam optimizer
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*6,validation_data=validation_generator,nb_val_samples=len(validation_samples)*6, nb_epoch=3)

model.save('model.h5')#Training model output