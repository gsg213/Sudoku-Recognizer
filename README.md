# Sudoku-Recognizer
An application in Python with OpenCV that digitize a Sudoku on a picture.


# Description
In this project OpenCV is used to detect the digits in an image from a Sudoku. The script uses KNN as ML model to clasify each new digit. The first thing is load the training dataset and labels of the text documents.

![My image](https://github.com/gsg213/Sudoku-Recognizer/blob/master/Img/image1.JPG)

Load the original image and and apply a bilateral filter to eliminate noise but keep most of the edges, the filter parameters were found by testing. I apply Findcontours to find all the contours within the image, comparing all contours and staying with the biggest, which is going to be the sudoku box. Approximate that contour to a square to obtain the 4 points of the corners of that contour.

![My image3](https://github.com/gsg213/Sudoku-Recognizer/blob/master/Img/image3.JPG)

Is used warpPerspective to cut out the contour and deform it to the size I choose. Findcontours is used again to detect all the contours within the sudoku, and through the cycle scroll through all the contours and calculate the areas of each contour and filter the areas between 120 and 600. Boundingrect function is used to obtain the coordinates and points w and h that indicate the width and height of the contour enclosed by a box. With a conditional filter the heights to leave out unwanted contours, with the values of then cut the new contour to pass it to the KNN model. The position of the image inside the grid is calculated in order to know the position of the digit and display the output.

![My image2](https://github.com/gsg213/Sudoku-Recognizer/blob/master/Img/imagen2.JPG)
