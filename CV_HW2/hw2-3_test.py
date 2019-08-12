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



test_img_path = sys.argv[1]
output_path = sys.argv[2]
filename_list = []

def loadfile():
	
	test_data = np.empty((10000,28,28))
	test_counter = 0
	
	for filename in os.listdir(test_img_path):
	
		filename_list.append(int(filename[0:4])) 
		
		image_read= misc.imread(test_img_path + filename) 
		test_data[test_counter] = image_read 
		test_counter += 1
	
	filename_list.sort()
	#print(filename_list)
	test_data = test_data.reshape(test_data.shape[0] , 28,28,1 ) #turn to 4 dim
	test_data = test_data /255.0
	
	return test_data
	

def outfile(result):
    outfile = open(output_path, 'w')
    leng = result.shape[0]
    ########################################################
    outfile.write("id,label\n")
    for i in range(leng):
        outfile.write(str(filename_list[i]))
        outfile.write(',')
        outfile.write(str(result[i]))
        outfile.write('\n')
    outfile.close()
    return

	
test_data = loadfile()


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

model.load_weights('hw2-3_weight.h5')

result = model.predict(test_data)
result = np.argmax(result,axis=1)
outfile(result)



