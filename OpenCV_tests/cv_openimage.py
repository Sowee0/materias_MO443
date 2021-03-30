import cv2 as cv 
import numpy as np

img = cv.imread("clone.jpg")

cv.imshow("Display Window", img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("clone.jpg", img)