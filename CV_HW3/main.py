import numpy as np
import cv2


# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
    #A = np.zeros((2*N, 8))
    # if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    counter = 0
    for i in range(N): # how many points
        x,y = u[i][0] , u[i][1]
        x_t , y_t = v[i][0] , v[i][1]
        A[i*2]  = ([x, y, 1, 0, 0, 0, -x_t*x, -x_t*y, -x_t])
        A[i*2+1]= ([0, 0, 0, x, y, 1, -y_t*x, -y_t*y, -y_t])
    #print(A)
    U, S, VT = np.linalg.svd((A.T).dot(A)) # h is the last column of V
    
    L = VT[-1,:] / VT[-1,-1]
    H = L.reshape(3,3)
    #print(H,'\n')
    return H


# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    #print(h, w)
    # TODO: some magic
    u = np.array([[0,0],[h -1, 0],[0, w-1 ],[h-1 , w-1 ]])
    H = solve_homography(u, corners)

    for j in range(h):
        for i in range(w):
            trans = H.dot([i,j,1]) #x'
            trans = trans / trans[-1]
            trans = np.round(trans[0:2]).astype(int)
            canvas[trans[1]][trans[0]] = img[j,i]
            
def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/wu.jpg')
    img2 = cv2.imread('./input/ding.jpg')
    img3 = cv2.imread('./input/yao.jpg')
    img4 = cv2.imread('./input/kp.jpg')
    img5 = cv2.imread('./input/lee.jpg')

    corners1 = np.array([[818, 352], [884, 352], [818, 407], [885, 408]])
    corners2 = np.array([[311, 14], [402, 150], [157, 152], [278, 315]])
    corners3 = np.array([[364, 674], [430, 725], [279, 864], [369, 885]])
    corners4 = np.array([[808, 495], [892, 495], [802, 609], [896, 609]])
    corners5 = np.array([[1024, 608], [1118, 593], [1032, 664], [1134, 651]])
    
    # TODO: some magic
    transform(img1, canvas, corners1)
    transform(img2, canvas, corners2)
    transform(img3, canvas, corners3)
    transform(img4, canvas, corners4)
    transform(img5, canvas, corners5)
    ##########################
    cv2.imwrite('part1.png', canvas)

    # Part 2
    img = cv2.imread('./input/screen.jpg')
    # TODO: some magic
    w,h,ch = img.shape
    W = 192 
    QR_distortion = np.array([[1040, 369], [1102, 395], [1036, 600], [982, 552]])
    QR_parallel   = np.array([[0 ,0 ], [W , 0], [W , W], [0 , W]])
    H_QR = solve_homography(QR_parallel ,  QR_distortion)
    
    MAP = np.zeros(shape=(W,W,3))
    for i in range(W):
        for j in range(W):
            temp = H_QR.dot([i,j,1])
            temp = temp / temp[-1]
            X,Y = np.round(temp[0:2]).astype(int)
            MAP[j][i] = img[Y,X]

    cv2.imwrite('part2.png', MAP)
    
    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    # TODO: some magic
    w,h,ch = img_front.shape
    #print(w,h)
    MAP = np.zeros(shape=(w,h,3)) #407 725 3
    #top_view_total = np.array([[62, 50], [659, 50], [659, 230], [62, 238]])
    top_view_total = np.array([[0,0],[h -1, 0],[0, w-1 ],[h-1 , w-1 ]])
    #front_view_total   = np.array([[135 , 163], [587 , 157], [659 , 230], [62 , 238]])
    front_view_total   = np.array([[122 , 112], [600 , 105], [1 , 336], [720 , 306]])
    H_view_total = solve_homography(top_view_total ,  front_view_total)
    
    for j in range(w):
        for i in range(h):
            temp = H_view_total.dot([i,j,1])
            temp = temp / temp[-1]
            X,Y = np.round(temp[0:2]).astype(int)
            #print(X,Y)
            #np.clip(Y, 0, 407)
            #np.clip(X, 0, 725)
            MAP[j][i] = img_front[Y,X]

    
    '''
    top_view_left = np.array([[62, 50], [109, 50], [109, 236], [62, 238]])
    front_view_left   = np.array([[135 , 163], [173 , 161], [109 , 236], [62 , 238]])
    H_view_left = solve_homography(front_view_left ,  top_view_left)
    
    top_view_middle = np.array([[345, 50], [380, 50], [386, 229], [340, 229]])
    front_view_middle   = np.array([[345 , 158], [380 , 158], [386 , 229], [340 , 229]])
    H_view_middle = solve_homography(front_view_middle ,  top_view_middle)
    '''
    #view_total = cv2.warpPerspective(img_front, H_view_total, (725, 407))
    #view_left = cv2.warpPerspective(img_front, H_view_left, (725, 407))
    #view_middle = cv2.warpPerspective(img_front, H_view_middle, (725, 407))
    #cv2.imwrite('part3.png', view_total)
    cv2.imwrite('part3.png', MAP)

    #cv2.imwrite('view_left.png', view_left)
    #cv2.imwrite('view_middle.png', view_middle)
    

if __name__ == '__main__':
    main()
   