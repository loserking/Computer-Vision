import numpy as np 
import os
import math
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE

whole_data_path = sys.argv[1]
test_image_path = sys.argv[2]
output_path     = sys.argv[3]

def average(image_array):
    average_f = np.mean(image_array, axis=0)
    average = np.round(average_f).astype(np.uint8)
    #im = Image.fromarray(average)
    #im.save('average_face.bmp')
    #im.show()
    return average_f


full_data = np.zeros( (400 , 56,46) ) 

index = 0
for i in range(1,41): #input filename example:  1_1.png 40_10.png
	for j in range(1,11):

		file_mame = whole_data_path +'/' +str(i) + '_' + str(j) +  '.png'
		image_read= misc.imread(file_mame) 
		full_data[index] = image_read 
		index += 1
		
full_data = full_data.reshape((400,56*46))          

test_image= misc.imread(test_image_path) 
test_image = test_image.reshape(56*46) 

averageFace = average(full_data)
X = full_data - averageFace #减去均值
x = test_image - averageFace
U, S, V = np.linalg.svd(X ,full_matrices=False)
weights = np.dot(V, x.T)

eigen_num = len(weights)

recon = averageFace + np.dot(weights[:eigen_num].T, V[:eigen_num, :])
img = Image.fromarray(recon.reshape(56,46).astype(np.uint8)) 
	
#mse_result = round(mse(img , test_image.reshape(56,46)) , 5)

img.save(output_path)