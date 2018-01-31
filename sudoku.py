###########Gustavo Andres Salazar Gomez - 2130286 #######################   

import cv2
import numpy as np
from numpy import array
from sklearn import datasets
import math


def corners(cnt):#rutina corners para organizar las 4 esquinas del sudoku en orden
    #esta rutina recibe c_ajustado que son las 4 esquinas del cuadrado del sudoku pero
    #ajustado a un cuadrado bien recto, para obtener solo 4 coordenadas
    
    #se crea una matriz de ceros para almacenar el nuevo orden
    cor = np.zeros((4,2),np.float32)

    #se suman los valores de cada coordenada, para esto se utiliza el comando sum
    #que permite la suma de cada celda
    mx=np.sum(cnt,axis=2)        

    #el valor minimo va a resultar en la esquina superior izquierda, que se asigna al 1
    #para que concuerde con el orden de los valores en el pts2
    cor[1] = cnt[np.argmin(mx)]
    #el valor maximo va a resultar en la esquina inferior derecha, y de igual manera se
    #asigna al 3 para conconrdar con el orden de pts2
    cor[3] = cnt[np.argmax(mx)]

    #el comando diff permite realizar una diferencia de los valores de cada celda
    mn=np.diff(cnt)
    #el valor minimo va a resultar en la esquina superior derecha, que se asigna al 0
    #para que concuerde con el orden de los valores en el pts2
    cor[0] = cnt[np.argmin(mn)]
    #el valor maximo va a resultar en la esquina inferior izquierda, que se asigna al 2
    #para que concuerde con el orden de los valores en el pts2
    cor[2] = cnt[np.argmax(mn)]

    #retorna el vector cor con todas las coordenadas organizadas de tal manera que encaje
    #en el getPerspective con los puntos del vector pts2
    return cor

def display_sudoku(mat):#esta rutina se encarga de imprimir en consola el sudoku en una rejilla
    
    for i in range(9):# i va a corresponder la posicion de las filas
        
        for j in range(9):# j va a corresponder la osicion de las columnas
                        
            if j == 2 or j == 5:# las posiciones seguidas de 2 y 5 llevaran el separador
                if mat[i,j] == 0:# si el valor actual es igual a cero imprime _ para dejarlo vacío
                    print '_',
                else:
                    print mat[i,j],#si el valor actual no es cero se imprime normalmente                
                print '|',#se imprimer el separador
            elif j == 8:# el ultimo valor de las columnas se imprime sin "," para que salte a una nueva linea
                if mat[i,j] == 0:# si el valor actual es igual a cero imprime _ para dejarlo vacío
                    print '_'
                else:
                    print mat[i,j]#si el valor actual no es cero se imprime normalmente                
            else:#si no es un vecino del borde o del separador imprime normalmente
                if mat[i,j] == 0:# si el valor actual es igual a cero imprime _ para dejarlo vacío
                    print '_',
                else:
                    print mat[i,j],#si el valor actual no es cero se imprime normalmente                               
                        
        if i == 2 or i == 5:#si ha pasado por la fila 2 o 5 imprime el separador
            print '------+-------+------'          
        
                

       

#se carga el data set de labels para el knn
labels = np.loadtxt('classifications.txt',np.float32)

#se cargan los datos de entrenamiento para la knn
data_images =np.loadtxt('flattened_images.txt',np.float32)

#reorganiza la info cargada en labels en un vector de 180 x 1 
labels= labels.reshape((labels.size, 1))
# se crea el k-nearest neighbors
knn=cv2.ml.KNearest_create()

#se pasan las imagenes de entranamiento, los labels respectivos y el parametro cv2.ml.ROW_SAMPLE
# que hace que considere la longitud del arreglo como 1 para toda la fila
knn.train(data_images, cv2.ml.ROW_SAMPLE, labels)
########## se incializan variables
img_w=20# ancho de la imagen para el resize
img_h=30# alto de la imagen para el resize
ssz=9# numero de cuadriculas
sz_imw = 504 #tamaño de la imagen para warp
se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))# elemento estructurante de tipo cuadrado o recto
mat_sudoku=np.zeros((9,9),np.uint8)# amtriz de ceros para ir almacenando los numeros reconosidos

    
original = cv2.imread('sudoku_final.jpg')#se carga la imagen original
original1=original.copy()# se realiza una copia de la original
original2=original.copy()# se realiza otra copia de la original
cv2.imshow('Original',original)
cv2.waitKey(0)
cv2.destroyAllWindows()


g = cv2.cvtColor(original,cv2.COLOR_BGR2GRAY)# se pasa a escala de grises para comenzar a trabajar la imagen
im = cv2.bilateralFilter(g,-1,35,3)# se le aplica un filtro bilateral para eliminar el ruido pero conservar
                                   #los bordes
cv2.imshow('gray',g)
cv2.imshow('Bilateral filter',im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# se realiza un threshold adaptativo sobre la imagen y se invierte con el parametro cv2.THRESH_BINARY_INV
imthresh = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,2)
dilat= cv2.dilate(imthresh,se,iterations = 1)# se le realiza una dilatacion a la imagen con el threshold
                                             # para aumentar los bordes, una sola iteracion

cv2.imshow('Threshold',imthresh)
cv2.imshow('dilatacion',dilat)
cv2.waitKey(0)
cv2.destroyAllWindows()

#se buscan los contornos sobre la imagen dilatada para poder encontrar el recuadro del sudoku
im2, contours, hc = cv2.findContours(dilat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# retorna contours que es un vector con las coordenadas que describen todos los contornos

# se incializan las variables de area, un acumulador y un contador de posicion
area=0# va a manetener momentaneamente el valor del area actual
acum=0# va a ir acumulando el valor de area mayor, manteniendo el mayor valor que encuentre
c_sudoku=0# va a contener el contorno con el mayor area, es decir el contorno del recuadro del sudoku

for i in range(len(contours)):# un ciclo for para recorrer todos los contornos y calcular sus areas 
                
    area=cv2.contourArea(contours[i])# calcula el area para el contorno actual
    if area > acum:# si el contorno actual es mayor al contorno acumulado, se reemplaza y pasa a ser
                        # el nuevo acumualdo
        acum = area# se asgina el nuevo acumulado
        c_sudoku=contours[i]# se asigna el contorno mas grande a medida que lo va hayando
            
            
#muestra el cuadrado mas grande y el cuadrado mejor ajustado para la transformacion
#de perspectiva
epsilon = 0.02*cv2.arcLength(c_sudoku,True)
c_ajustado = cv2.approxPolyDP(c_sudoku,epsilon,True)#se aproxima un nuevo cuadrado ajustado al cuadrado hallado

cv2.drawContours(original1,[c_sudoku],0,(100,135,135),2)# se dibuja el contorno encontrado en original1
cv2.drawContours(original2,[c_ajustado],0,(150,0,135),2)# se dibuja el contorno ajustado en original1
cv2.imshow('contornos',original1)
cv2.imshow('contorno ajustado',original2)
cv2.waitKey(0)
cv2.destroyAllWindows()

corn = corners(c_ajustado)# se envia c_ajustado para organizar las esquinas en orden, recibiendo corn

#se asignan los puntos a donde se van a transformar la perspectiva del recuadro del sudoku
#el orden esta establecido como esquina superior derecha, superior izquierda, inferior izquierda e
#inferior derecha respectivamente, dandole un tamaño final de 504x504
pts2 = np.array([[504,0],[0,0],[0,504],[504,504]],np.float32)

#se calculan los valores de la transformacion de perspectiva que van a ser usados en el warp
pers = cv2.getPerspectiveTransform(corn,pts2)

# se realiza el warp de la imagen dilatada al principio, con los datos calculados del perspective
# y dando un tamaño final de 504x504
warp1 =cv2.warpPerspective(dilat, pers,(504,504))
#se realiza una erocion despues del warp para disminuir el area de los contornos dentro de la imagen 
warp= cv2.erode(warp1,se,iterations = 1)

cv2.imshow('warpPers',warp1)
cv2.imshow('Erode warpPers',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# se hallan de nuevo los contornos pero ahora dentro del recuadro del sudoku, para comenzar a detectar
# los numeros contenidos
im_1 , contours2, hc2 = cv2.findContours(warp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
area2=0 
for i in range(len(contours2)): 

    area2=cv2.contourArea(contours2[i])#se calcula el area de todos los contornos internos   
        
    if 120 < area2 and area2 < 600 :#se filtran las areas para solo dejar pasar las areas de los numeros
                                    #estos valores se hallan realizando pruebas hasta conseguir todos los
                                    #digitos
        
        x,y,w,h = cv2.boundingRect(contours2[i])#se aproxima un cuadrado mas proximo al contorno, sin tener
                                                #en cuenta la rotacion la rotacion, teniendo como respuesta
                                                #las coordenadas X y Y, y entrega el ancho y alto del contorno

        if h < 74 and h > 18:# se filtran las alturas del boundingrect para solo dejar pasar los digitos
                             # y dejar por fuera figuras indeseadas
                             
            dig_rec1 = warp[y:y+h,x:x+w]#se recortan los digitos con los datos obtenidos del boundingrect x,y,w,h 
            
            dig_rec = cv2.resize(dig_rec1, (img_w, img_h))#se hace un resize de la imagen recortada a los tamaños definidos

            #se reorganiza la imagen como un vextor de 1x600 que es el numero total de pixeles de la imagen
            dig_rec= dig_rec.reshape((1,img_w*img_h))

            #se convierte a float32 la imagen reorganizada
            dig_rec = np.float32(dig_rec)

            #se envia la informacion del digito encontrado en forma de vector a la knn
            #se agrega un parametro de k=1 para que encuentre el vecino mas cercano posible
            #retorna los vecinos mas cercanos, la distancia mas cercana con el vecino mas proximo
            #y el resultado de la prediccion en result
            ret, result, neighbours, dist = knn.findNearest(dig_rec, k = 1)
                      
            #convierte el resultado en entero y luego en caracter debido a que la salida esta codificada,
            #puesto que es capaz de reconocer letras tambien por el data set encontrado
            result = chr(int(result[0][0]))
            
            #se imprime el resultado 
            print result
            
            cv2.imshow('digit',dig_rec1)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            
            
            #se calcula la posicion del digito encontrado dentro de la rejilla del sudoku,
            #por lo tanto se buscan las coordenadas X y Y, se le suma w y h divido entre 2 para encontrar el centro
            # del digito, y por ultimo se divide entre 56 puesto que el tamaño total de la imagen(504) dividido entre
            # el numero de cuadriculas (9) da como resultado (504/9=56), esta divicion entre 56 nos da la posicion dentro
            #de los 9 cuadros en el sudoku
            posx = (x+(w/2))/56
            posy= (y+(h/2))/56          

            iposy=int(posy)# las posiciones obtenidas se vuelven enteros
            iposx =int(posx) 

            #se reemplaza en la posicion encontrada el resultado devuelto en result por la knn
            mat_sudoku[iposy,iposx]=result
            

#se envia la matriz encontrada a la funcion de display_sudoku para escribirla en consola
display_sudoku(mat_sudoku)

#se imprime una imagen final desde el warp para comparar con la informacion imprimida en consola
cv2.imshow('Final',warp)
cv2.waitKey(0)
cv2.destroyAllWindows()
