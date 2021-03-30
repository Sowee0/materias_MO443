import numpy as np
import cv2 as cv

#Iniciando o objeto do dispositivo de captura
cap = cv.VideoCapture(0,cv.CAP_DSHOW)

#verificando se o mesmo está aberto
if not cap.isOpened():
    print("Couldn't open video device")
    exit()

#lendo os frames
while True:
    ret, frame = cap.read()
    cv.ellipse(frame,(256,256),(100,50),0,0,180,255,-1)

    if not ret:
        print("Couldn't read frame. Bad stream? Ending...")
        break
    
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv.imshow('frame', gray)
    if cv.waitKey(1) == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
