import numpy as np
import random
import math
import h5py
import sys
import os
from PIL import Image
from scipy import misc
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import AveragePooling2D

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



path = sys.argv[1]
#valid_path = sys.argv[2]


def loadfile():
    print('loading file')
    train_data = np.empty((50000,28,28))
    train_label = np.zeros((50000,), dtype=np.int)
    
    valid_data  = np.empty((10000,28,28))
    valid_label = np.zeros((10000,), dtype=np.int)
    
    train_counter = 0
    Valid_counter = 0
    
    for i in range(10): 
        train_number = path +'train/class_' + str(i) 
        valid_number = path +'valid/class_' + str(i) 
        
        #trian_set
        for filename in os.listdir(train_number):
            image_read= misc.imread(train_number + '/' +filename) 
            train_data[train_counter] = image_read 
            train_label [train_counter] = i
            train_counter += 1
            
        #valid_set
        for filename in os.listdir(valid_number):
            image_read= misc.imread(valid_number + '/' + filename) 
            valid_data[Valid_counter] = image_read 
            valid_label [Valid_counter] = i
            Valid_counter += 1
    
    train_data = train_data.reshape(train_data.shape[0] , 28,28,1 ) #turn to 4 dim
    valid_data = valid_data.reshape(valid_data.shape[0] , 28,28,1 ) #turn to 4 dim
    #row = pixel.shape[0]
    #col = pixel.shape[1]
    
    train_data = train_data /255.0
    valid_data = valid_data /255.0
    
    train_label = np_utils.to_categorical(train_label, 10)
    valid_label = np_utils.to_categorical(valid_label, 10)
    

    
    return train_data, train_label , valid_data ,valid_label



train_data, train_label , valid_data ,valid_label = loadfile()

print(train_data.shape)
print(train_label.shape)
print(valid_data.shape)
print(valid_label.shape)

#Model
model = Sequential()

model.add(Convolution2D(16,(3,3),input_shape=(28,28,1) , padding='same'))
model.add(Convolution2D(32,(3,3),activation='relu' , padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64,(3,3), padding='same'))
model.add(Convolution2D(128,(3,3),activation='relu', padding='same'))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=256,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(units=10,activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',optimizer= 'Nadam', metrics=['accuracy'])

batch_size = 1000
num_classes = 10
epochs = 5
history = model.fit(train_data, train_label,
                    batch_size= batch_size,
                    epochs= epochs,
					shuffle=True,
                    verbose=1,
                    validation_data=(valid_data, valid_label))

score = model.evaluate(train_data,train_label,batch_size=batch_size)

print('\nTrain Acc:', score[1] )

model.save_weights('hw2-3_weight_test.h5')




