
# coding: utf-8

# In[ ]:

import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import scipy.misc
import sys

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3], [0.299, 0.587, 0.114])

sigma_s_L = [1 , 2 , 3]
sigma_r_L = [0.05 , 0.1 , 0.2]
candidate_vote = np.zeros(66)
JBF_LIST= []

'sigma_s = 1'
'sigma_r = 0.05'
'window_size = 3*sigma_s'
name = sys.argv[1]
img1a = np.asarray(Image.open(name))

list_num = [ 0, 0.1, 0.2 , 0.3 , 0.4, 0.5 , 0.6 ,0.7 ,0.8 ,0.9, 1]
candidate_list = []
for i in list_num:
    for j in list_num:
        for k in list_num:
            if(round((i+j+k),4) == 1.0):
                candidate_list.append((i,j,k))

'print(len(candidate_list))'

'generate 66 candidate for each picture'
img1a_candidate = []
'img1b_candidate = []'
'img1c_candidate = []'
for tup in candidate_list:
    img1a_candidate.append(np.dot(img1a[:,:,:3], tup))
    'img1b_candidate.append(np.dot(img1b[:,:,:3], tup))'                      
    'img1c_candidate.append(np.dot(img1c[:,:,:3], tup))'
    
img1a_candidate = np.asarray(img1a_candidate)
'img1b_candidate = np.asarray(img1b_candidate)'
'img1c_candidate = np.asarray(img1c_candidate)'

candidate_num , length , width = img1a_candidate.shape
JBF_ans = np.zeros(shape=(length , width, 3))
bilateral_ans = np.zeros(shape=(length , width, 3))

img1a_candidate = img1a_candidate / 255
'img1b_candidate = img1b_candidate / 255'
'img1c_candidate = img1c_candidate / 255'

img1a = img1a / 255
'img1b = img1b / 255'
'img1c = img1c / 255'
'len(candidate_list)'
for sigma_s in sigma_s_L:
    window_size = 3*sigma_s
    for sigma_r in sigma_r_L: 
        for player_num in range(0,66):
            JBF_ans = np.zeros(shape=(length , width, 3))
            for i in range(0,length):
                for j in range(0 , width):
                    'return to 0 for every pixel'
                    Fnorm = 0
                    out_accumulate_r = 0
                    out_accumulate_g = 0
                    out_accumulate_b = 0

                    'Generate the window of the kernel'
                    imin=max(i-window_size,0)
                    imax=min(i+window_size,length-1)
                    jmin=max(j-window_size,0)
                    jmax=min(j+window_size,width-1)

                    'RGB 3 channel'
                    I_r=img1a[imin:imax , jmin:jmax, 0]
                    I_g=img1a[imin:imax , jmin:jmax, 1]
                    I_b=img1a[imin:imax , jmin:jmax, 2]

                    range_filter_r=np.exp(-((I_r-img1a_candidate[player_num, imin:imax , jmin:jmax])**2)/(2*(sigma_r**2)))
                    range_filter_g=np.exp(-((I_g-img1a_candidate[player_num, imin:imax , jmin:jmax])**2)/(2*(sigma_r**2)))
                    range_filter_b=np.exp(-((I_b-img1a_candidate[player_num, imin:imax , jmin:jmax])**2)/(2*(sigma_r**2)))

                    Gr = range_filter_r * range_filter_g * range_filter_b

                    'compute Gs and the result'
                    Gs = np.zeros(shape=( imax-imin, jmax-jmin ))

                    for kernel_i in range(imin , imax):
                        for kernel_j in range(jmin , jmax):
                            Gs[kernel_i-imin][kernel_j-jmin] = math.exp( -((kernel_i-i)**2 + (kernel_j-j)**2) /(2*sigma_s**2) )

                    Fnorm = np.sum(Gs*Gr)
                    out_accumulate_r = np.sum(Gs * Gr * img1a[imin:imax , jmin:jmax ,0])
                    out_accumulate_g = np.sum(Gs * Gr * img1a[imin:imax , jmin:jmax ,1])
                    out_accumulate_b = np.sum(Gs * Gr * img1a[imin:imax , jmin:jmax ,2])


                    'JBF result of single pixel'
                    JBF_ans[i][j][0] =  out_accumulate_r / Fnorm
                    JBF_ans[i][j][1] =  out_accumulate_g / Fnorm  
                    JBF_ans[i][j][2] =  out_accumulate_b / Fnorm

                    'print( Gr.shape , Gs.shape , (imin,imax),(jmin,jmax))'
            JBF_LIST.append(JBF_ans)
            '''
            scipy.misc.imsave('JbF_img1c_s' + str(sigma_s)+'r'+ str(sigma_r) + '.png' , JBF_ans)
            'print(player_num)'
            '''
        'JBF = np.asarray(JBF_LIST)'
        JBF = np.empty(shape=(66 , length , width, 3))
        for zz in range(0,66):
            JBF[zz] =  np.asarray(JBF_LIST[zz])
            
        '''
        print('done Joint Bilteral Filter')
        '''
        
        'bilateral filter'
        for i in range(0,length):
            for j in range(0 , width):
                'return to 0 for every pixel'
                Bi_Fnorm = 0
                Bi_out_accumulate_r = 0
                Bi_out_accumulate_g = 0
                Bi_out_accumulate_b = 0

                'Generate the window of the kernel'
                imin=max(i-window_size,0)
                imax=min(i+window_size,length-1)
                jmin=max(j-window_size,0)
                jmax=min(j+window_size,width-1)

                'RGB 3 channel'
                Bi_I_r=img1a[imin:imax , jmin:jmax, 0]
                Bi_I_g=img1a[imin:imax , jmin:jmax, 1]
                Bi_I_b=img1a[imin:imax , jmin:jmax, 2]

                Bi_range_filter_r=np.exp(-((Bi_I_r-img1a[imin:imax , jmin:jmax , 0])**2)/(2*(sigma_r**2)))
                Bi_range_filter_g=np.exp(-((Bi_I_g-img1a[imin:imax , jmin:jmax , 1])**2)/(2*(sigma_r**2)))
                Bi_range_filter_b=np.exp(-((Bi_I_b-img1a[imin:imax , jmin:jmax , 2])**2)/(2*(sigma_r**2)))

                Bi_Gr = Bi_range_filter_r * Bi_range_filter_g * Bi_range_filter_b

                'compute Gs and the result'
                Bi_Gs = np.zeros(shape=( imax-imin, jmax-jmin ))

                for kernel_i in range(imin , imax):
                    for kernel_j in range(jmin , jmax):
                        Bi_Gs[kernel_i-imin][kernel_j-jmin] = math.exp( -((kernel_i-i)**2 + (kernel_j-j)**2) /(2*sigma_s**2) )

                Bi_Fnorm = np.sum(Bi_Gs*Bi_Gr)
                Bi_out_accumulate_r = np.sum(Bi_Gs * Bi_Gr * img1a[imin:imax , jmin:jmax ,0])
                Bi_out_accumulate_g = np.sum(Bi_Gs * Bi_Gr * img1a[imin:imax , jmin:jmax ,1])
                Bi_out_accumulate_b = np.sum(Bi_Gs * Bi_Gr * img1a[imin:imax , jmin:jmax ,2])


                'BF result of single pixel'
                bilateral_ans[i][j][0] =  Bi_out_accumulate_r / Bi_Fnorm
                bilateral_ans[i][j][1] =  Bi_out_accumulate_g / Bi_Fnorm  
                bilateral_ans[i][j][2] =  Bi_out_accumulate_b / Bi_Fnorm

        '''
        scipy.misc.imsave('bilateral_test_1c' + str(sigma_s)+'r'+ str(sigma_r) + '.png' , bilateral_ans)
        print('done basic bilteral filter')
        '''
        Cost = np.empty(66)
        for index in range(0,66):
            Cost[index] = np.sum(abs(JBF[index]-bilateral_ans))
        np.savetxt('Cost_'+ str(sigma_s)+'r'+ str(sigma_r)+ '.txt' , Cost, fmt='%f')
        'compute local min'
        'corner index 0 , 10 , 65'
        local_min_ind = np.where(Cost == min(Cost[0],Cost[1],Cost[11]))
        candidate_vote[local_min_ind[0][0]] += 1

        local_min_ind = np.where(Cost == min(Cost[10],Cost[9],Cost[20]))
        candidate_vote[local_min_ind[0][0]] += 1

        local_min_ind = np.where(Cost == min(Cost[65],Cost[64],Cost[63]))
        candidate_vote[local_min_ind[0][0]] += 1

        'side1'
        for ind in range(1,9):
            local_min_ind = np.where(Cost == min(Cost[ind],Cost[ind+1],Cost[ind-1],Cost[ind+10],Cost[ind+10],Cost[ind+11]))
            candidate_vote[local_min_ind[0][0]] += 1

        'side2'  
        side2=[20,29,37,44,50,55,59,62,64]
        level=10
        for ind in side2:
            local_min_ind = np.where(Cost == min(Cost[ind-level],Cost[ind+level-1],Cost[ind-1],Cost[ind-level-1]))
            candidate_vote[local_min_ind[0][0]] += 1
            level -= 1

        'side3'
        side3=[11,21,30,38,45,51,56,60,63]
        level=10
        for ind in side3:
            local_min_ind = np.where(Cost == min(Cost[ind+level],Cost[ind+1],Cost[ind-level],Cost[ind-level-1]))
            candidate_vote[local_min_ind[0][0]] += 1
            level -= 1

        'middle'
        level=10
        counter=0
        ind = 12
        while(ind<62):
            local_min_ind = np.where(Cost == min(Cost[ind+1],Cost[ind-1],Cost[ind-level],Cost[ind-level-1],Cost[ind+level],Cost[ind+level+1]))
            candidate_vote[local_min_ind[0][0]] += 1
            counter += 1
            if(counter == level-2):
                ind +=2
                counter = 0
                level -= 1
            ind+=1

        'print(candidate_vote)'

img1a = np.asarray(Image.open(name))
top_3_idx = np.argsort(candidate_vote)[-3:]
top_3_values = [candidate_vote[i] for i in top_3_idx]
'print(top_3_idx)'
for number, ind in enumerate(reversed(top_3_idx)):
    ans_gray1a = np.dot(img1a[:,:,:3], candidate_list[ind])
    print(number,ind)
    print(candidate_list[ind])
    scipy.misc.imsave(str(name[0:2]) + '_y' + str(number+1) +'.png', ans_gray1a)

