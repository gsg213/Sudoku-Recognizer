###########Gustavo Andres Salazar Gomez#######################   

import cv2
import numpy as np
from numpy import array
from sklearn import datasets
import math


def corners(cnt):
    
    cor = np.zeros((4,2),np.float32)
 
    mx=np.sum(cnt,axis=2)        

    cor[1] = cnt[np.argmin(mx)]
    cor[3] = cnt[np.argmax(mx)]    
    mn=np.diff(cnt)
    
    cor[0] = cnt[np.argmin(mn)]
    cor[2] = cnt[np.argmax(mn)]
    
    return cor

def display_sudoku(mat):
    
    for i in range(9):
        
        for j in range(9):
                        
            if j == 2 or j == 5:
                if mat[i,j] == 0:
                    print '_',
                else:
                    print mat[i,j],         
                print '|',
            elif j == 8:
                if mat[i,j] == 0:
                    print '_'
                else:
                    print mat[i,j]               
            else:
                if mat[i,j] == 0:
                    print '_',
                else:
                    print mat[i,j],                               
                        
        if i == 2 or i == 5:
            print '------+-------+------'          
        

labels = np.loadtxt('classifications.txt',np.float32)

data_images =np.loadtxt('flattened_images.txt',np.float32)

labels= labels.reshape((labels.size, 1))

knn=cv2.ml.KNearest_create()

knn.train(data_images, cv2.ml.ROW_SAMPLE, labels)

img_w=20
img_h=30
ssz=9
sz_imw = 504 
se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
mat_sudoku=np.zeros((9,9),np.uint8)
    
original = cv2.imread('sudoku_final.jpg')
original1=original.copy()
original2=original.copy()
cv2.imshow('Original',original)
cv2.waitKey(0)
cv2.destroyAllWindows()

g = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)
im = cv2.bilateralFilter(g,-1,35,3)

cv2.imshow('gray',g)
cv2.imshow('Bilateral filter',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

imthresh = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)
dilat= cv2.dilate(imthresh,se,iterations = 1)

cv2.imshow('Threshold',imthresh)
cv2.imshow('dilatacion',dilat)
cv2.waitKey(0)
cv2.destroyAllWindows()

im2, contours, hc = cv2.findContours(dilat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

area=0
acum=0
c_sudoku=0

for i in range(len(contours)):
                
    area=cv2.contourArea(contours[i])
    if area > acum:
        acum = area
        c_sudoku=contours[i]
            
epsilon = 0.02*cv2.arcLength(c_sudoku,True)
c_ajustado = cv2.approxPolyDP(c_sudoku,epsilon,True)

cv2.drawContours(original1,[c_sudoku],0,(100,135,135),2)
cv2.drawContours(original2,[c_ajustado],0,(150,0,135),2)
cv2.imshow('contornos',original1)
cv2.imshow('contorno ajustado',original2)
cv2.waitKey(0)
cv2.destroyAllWindows()

corn = corners(c_ajustado)

pts2 = np.array([[504,0],[0,0],[0,504],[504,504]],np.float32)

pers = cv2.getPerspectiveTransform(corn,pts2)

warp1 =cv2.warpPerspective(dilat, pers,(504,504))

warp= cv2.erode(warp1,se,iterations = 1)

cv2.imshow('warpPers',warp1)
cv2.imshow('Erode warpPers',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

im_1 , contours2, hc2 = cv2.findContours(warp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
area2=0 
for i in range(len(contours2)): 

    area2=cv2.contourArea(contours2[i])
        
    if 120 < area2 and area2 < 600 :
        x,y,w,h = cv2.boundingRect(contours2[i])
        if h < 74 and h > 18:
                             
            dig_rec1 = warp[y:y+h,x:x+w]
            
            dig_rec = cv2.resize(dig_rec1, (img_w, img_h))
            
            dig_rec= dig_rec.reshape((1,img_w*img_h))
            
            dig_rec = np.float32(dig_rec)

            ret, result, neighbours, dist = knn.findNearest(dig_rec, k = 1)
            result = chr(int(result[0][0]))

            print result
            
            cv2.imshow('digit',dig_rec1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            posx = (x+(w/2))/56
            posy= (y+(h/2))/56          

            iposy=int(posy)
            iposx =int(posx) 

            mat_sudoku[iposy,iposx]=result

display_sudoku(mat_sudoku)

cv2.imshow('Final',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
