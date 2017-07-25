import csv
import matplotlib.image as mpimg
import numpy as np        
import os

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:]

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                measurement = float(batch_sample[3])
                
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/' + filename
                    image = mpimg.imread(current_path)
                    images.append(image)
                    
                    # i==0: center img, i==1: left img, i==2: right img, add correction for left and right images
                    if i==1:
                        measurement = measurement + 0.2
                    elif i==2:
                        measurement = measurement - 0.2
        
                    measurements.append(measurement)
            
            # Flip
            augmented_images = []
            augmented_measurements = []
            
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_images.append(np.fliplr(image))
                augmented_measurements.append(measurement)
                augmented_measurements.append(-1.0*measurement)
                
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D

model = Sequential()
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape= (160, 320,3)))
model.add(Lambda(lambda x: x/127.5 - 1.0))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 
                 len(train_samples)*6, validation_data=validation_generator, 
                 nb_val_samples=len(validation_samples)*6, nb_epoch=3)

model.save('model.h5')