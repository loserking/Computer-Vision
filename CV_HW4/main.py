import numpy as np
import cv2
import time

window = 6
def zncc(img1, img2):
    std1 = np.std(img1)
    std2 = np.std(img2)
    avg1 = np.mean(img1)
    avg2 = np.mean(img2)
    s = np.sum((img1 - avg1) * (img2- avg2) )
    
    return s / ( ((2*window+1)**2) *std1 * std2 )

def ssd(img1, img2):
    ssd = np.sum(np.sqrt((img1 - img2)**2))
    
    return float(ssd)
    
def computeDisp(Il, Ir, max_disp):
	h, w, ch = Il.shape
	labels = np.zeros((h, w), dtype=np.uint8)
	labels_left = np.zeros((h, w), dtype=np.float)
	labels_right = np.zeros((h, w), dtype=np.float)
	
	MAP_left = np.zeros((h, w , max_disp ), dtype=np.float32)
	MAP_right = np.zeros((h, w , max_disp ), dtype=np.float32)
	
	gray_img_L = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
	gray_img_R = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
	
	#padding if use window style cost
	#gray_img_L = cv2.copyMakeBorder(gray_img_L, window , window , window+max_disp , window+max_disp ,cv2.BORDER_REFLECT_101) 
	#gray_img_R = cv2.copyMakeBorder(gray_img_R, window , window , window+max_disp , window+max_disp ,cv2.BORDER_REFLECT_101)

	gray_img_L = cv2.copyMakeBorder(gray_img_L, 0 , 0 , max_disp , max_disp ,cv2.BORDER_REFLECT_101) 
	gray_img_R = cv2.copyMakeBorder(gray_img_R, 0 , 0 , max_disp , max_disp ,cv2.BORDER_REFLECT_101)

	sobel_right = cv2.Sobel(gray_img_R,cv2.CV_64F,1,0, ksize=3)
	sobel_left = cv2.Sobel(gray_img_L,cv2.CV_64F,1,0, ksize=3)
	alpha = 0.9

	#plt.imshow(sobel_left,cmap='gray')
	#plt.show()

	#print('original h , w :', h,w)

	#print('after padding h , w = : ', gray_img_L.shape)

	# >>> Cost computation
	tic = time.time()
	# TODO: Compute matching cost from Il and Ir

	for i in range(0,h):
		#print(".", end="", flush=True) #each row
		for j in range( max_disp , w+max_disp):

			#left_rigion = gray_img_L[i-window : i+window , j-window : j+window ] #reference

			for disp in range(max_disp):
				
				MAP_left[i , j-max_disp , disp] = \
						(1-alpha)*( abs( float(gray_img_L[i, j])- float( gray_img_R[i, j-disp] ) ) )\
						+ (alpha)*( abs( float(sobel_left[i, j])-float(sobel_right[i, j-disp]) ) )
				
				#right_rigion = gray_img_R[ i-window : i+window , j-window-disp : j+window-disp ] #compare
				#MAP_left[i-window , j-window-max_disp , disp] = zncc(left_rigion , right_rigion )
				
				#MAP_left[i-window , j-window-max_disp , disp] = ssd(left_rigion , right_rigion )  
				

	toc = time.time()
	print('* Elapsed time (cost computation): %f sec.' % (toc - tic))
	
    # >>> Cost aggregation
	tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
	for d in range(max_disp):
		#MAP_left[:,:,d] = cv2.bilateralFilter(MAP_left[:,:,d],6,75,75)
		#MAP_left[:,:,d] = cv2.boxFilter(MAP_left[:,:,d],cv2.CV_64F,(5,5))
		MAP_left[:,:,d] = cv2.ximgproc.guidedFilter(Il,MAP_left[:,:,d],3,75)
	toc = time.time()
	print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))

	# >>> Disparity optimization
	tic = time.time()
	# TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
	for i in range(h):
		for j in range(w):
			#labels_left[i,j] = np.argmax(MAP_left[i,j,:])
			labels_left[i,j] = np.argmin(MAP_left[i,j,:]) #for ssd
			
	toc = time.time()
	print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
	tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
	
	#print('refinement')
	#print('#Left-right consistency check ')
	#Left-right consistency check 
	for i in range(0,h):
		#print(".", end="", flush=True) #each row
		for j in range( max_disp , w+max_disp):

			#left_rigion = gray_img_L[i-window : i+window , j-window : j+window ] #reference

			for disp in range(max_disp):
				
				MAP_right[i , j-max_disp , disp] = \
						(1-alpha)*( abs( float(gray_img_L[i, j+disp])- float( gray_img_R[i, j] ) ) )\
						+ (alpha)*( abs( float(sobel_left[i, j+disp])- float(sobel_right[i, j]) ) )
				
				
	#aggregation
	for d in range(max_disp):
		#MAP_right[:,:,d] = cv2.bilateralFilter(MAP_right[:,:,d],6,75,75)
		#MAP_right[:,:,d] =cv2.boxFilter(MAP_right[:,:,d],cv2.CV_64F,(5,5))
		MAP_right[:,:,d] = cv2.ximgproc.guidedFilter(Ir,MAP_right[:,:,d],3,75)
	# WTA
	for i in range(h):
		for j in range(w):
			#labels_right[i,j] = np.argmax(MAP_right[i,j,:])
			labels_right[i,j] = np.argmin(MAP_right[i,j,:]) #for ssd
				
    #build hole_mask
	labels_hole_mask = np.zeros((h, w))
	pre_labels = np.zeros((h, w), dtype=np.uint8)
	for i in range(h):
		for j in range(w):
			if labels_right[i, j- int(labels_left[i,j])] != labels_left[i,j] :
				labels_hole_mask[i,j] = 1
			else:             
				pre_labels[i,j] = labels_left[i,j] #note that label answer might be 0

    #print('seems correct value')       
    #plt.imshow(pre_labels*4,cmap='gray')
    #plt.show()

    #Left-right consistency check ends

    # hole filling
    #fill left-most and right-most pixel by the first and last nonzero value
	#print('# hole filling')
	for i in range(h):
		for j in range(w):
			if labels_hole_mask[i,j] == 1: #hole (not consistant answer then fill with nearby value)
				#find the nearest value that are not hole
				for look_right in range(w-j):
					if labels_hole_mask[i,j+look_right] == 0 : #not hole
						break

				for look_left in range(j+1):
					if labels_hole_mask[i,j-look_left] == 0: #not hole
						break
				if look_right > look_left: #choose nearest
					pre_labels[i,j] = labels_left[i,j-look_left]
				else:
					pre_labels[i,j] = labels_left[i,j+look_right]
					
    #print('after hole filling')
    #plt.imshow(pre_labels*4,cmap='gray')
    #plt.show()

    #cv2.imwrite('cones_left.png', np.uint8(labels_left * 4))''''''

	#print('#WMF')
    #for it in range(3):
	labels = cv2.ximgproc.weightedMedianFilter(Il,pre_labels,11)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,11)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,11)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,9)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,9)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,7)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,3)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,3)
	labels = cv2.ximgproc.weightedMedianFilter(Il,labels,3)
    #plt.imshow(labels*4,cmap='gray')
    #plt.show()
	
	toc = time.time()
	print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))

	return labels


def main():
	
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))
	
    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))

    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))


if __name__ == '__main__':
    main()
