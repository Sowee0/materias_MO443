import numpy as np
import cv2 as cv

img = np.zeros([512, 512, 3], np.int8)

cv.line(img,(0,0),(511,511),(255,0,0),5)
cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
cv.ellipse(img,(256,256),(100,50),0,0,180,255,-1)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv.LINE_AA)

cv.imshow("Imagem", img)
k = cv.waitKey(0)


