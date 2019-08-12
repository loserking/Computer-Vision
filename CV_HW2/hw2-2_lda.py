
# coding: utf-8

# In[158]:

import numpy as np 
import os
import math
from scipy import misc
from PIL import Image
import matplotlib.pyplot as plt
import sys
from sklearn.manifold import TSNE

train_N = 280
test_N = 120
C=40

input_path = sys.argv[1]
output_path = sys.argv[2]

def average(image_array):
    average_f = np.mean(image_array, axis=0)
    average = np.round(average_f).astype(np.uint8)
    
    return average_f

def train_avg_class(training_set , NsubC_dim):
    avgface_class = np.zeros( (40 , NsubC_dim) ) 

    for i in range(40):
        avgface_class[i] = average(training_set[i*7:(i+1)*7])#.reshape(56,46)
        #plt.imshow(avgface_class[i], cmap='gray')
        #plt.show()
        
    return avgface_class

def PCA(data , average):
    mu = data - average
    U, S, VT = np.linalg.svd(mu ,full_matrices=False)

    #output eigenfaces
    #print(U.shape)
    #print(S.shape)
    #print(VT.shape)
    V =VT.T
    
    return V

def readfile():
    training_set = np.zeros( (280 , 56,46) ) 
    testing_set = np.zeros( (120 , 56,46) ) 

    index_train = 0
    index_test  = 0
    for i in range(1,41): #input filename example:  1_1.png 40_10.png
        for j in range(1,11):

            file_mame = input_path +'/'+str(i) + '_' + str(j) +  '.png'
            image= misc.imread(file_mame) 
            if j > 7:
                testing_set[index_test] = image 
                index_test += 1
            else:
                training_set[index_train] = image 
                index_train += 1
                
    training_set = training_set.reshape((280,56*46)) 
    testing_set  = testing_set.reshape((120,56*46))

    return training_set , testing_set


if __name__ == "__main__":
    print('main')
    
    training_set , testing_set = readfile()
    trainset_averageFACE = average(training_set)
    test_averageFACE  = average(testing_set)
    X_train = training_set - trainset_averageFACE
     
    train_EigenFace = PCA(training_set , trainset_averageFACE )
    test_EigenFace = PCA(testing_set , test_averageFACE )
    print('train_EigenFace :' , train_EigenFace.shape )
    
    train_PCA_useful_eigen = train_EigenFace[:, 0:(train_N - C)]
    print('PCA_useful_eigen :' , train_PCA_useful_eigen.shape)
    
    train_projected_data = np.dot( training_set , train_PCA_useful_eigen)
    print('PCA_projected_data :',train_projected_data.shape) #dim = 280*240
    
    train_avgface_class = train_avg_class(train_projected_data , train_projected_data.shape[1])
    print('avg_class: ',train_avgface_class.shape) #dim = 40*240
    
    train_avgface_all = average(train_projected_data)
    print('avg_glb : ',train_avgface_all.shape) #dim = 240
    #test_avgface_all  = average(testing_set)
    
    SB = 7 * np.dot( (train_avgface_class - train_avgface_all).T , ((train_avgface_class - train_avgface_all)))

    print('SB: ' , SB.shape)
    
    SW = np.zeros((240,240))
    
    for i in range(C):
        X = train_projected_data - train_avgface_class[i]
        SW += np.dot(X.T , X)
        
    print('SW: ' , SW.shape)

    e_vals, e_vecs = np.linalg.eig(np.linalg.inv(SW).dot(SB))
    e_vecs = e_vecs.real
   
    
    e_vecs = e_vecs[: , :C-1]
    #print(e_vecs)
    #print(e_vecs.shape)
    #print('Eigenvectors \n%s' %e_vecs)
    #print('\nEigenvalues \n%s' %e_vals)
    
    fisherface = np.dot(  train_PCA_useful_eigen , e_vecs )
    
    fisherface = fisherface.T
    output = fisherface[0].reshape(56,46)
    misc.imsave(output_path ,output)
    
    
    '''
    #plt top five fisherfaces
    fig = plt.figure(figsize=(9, 9))
    for i in range(5):
        img = fisherface[i].reshape(56,46)
        ax = fig.add_subplot(1, 5, i+1)
        plt.xlabel('fisherface: ' +str(i))
        ax.imshow(img, cmap='gray')
        plt.yticks(np.array([]))
        plt.xticks(np.array([]))

    #plt.tight_layout()
    plt.show()
    ''' 
    
    
    print('over')





