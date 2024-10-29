import cv2
import numpy as np
from time import sleep
import imutils

minimum_width=80 # Minimum width of the rectangle
minimum_height=80 # Minimum height of the rectangle


offset=6 # Error allowed between pixel

counting_line=550 # Counting line position

delay= 60 # Video FPS

detec = []
cars= 0


def paste_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

# cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(1)
subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    ret , frame1 = cap.read()
    frame1=imutils.resize(frame1,width=1200)
    tempo = float(1/delay)
    sleep(tempo)
    grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(9,9),5)
    img_sub = subtracao.apply(blur)
    # wide open
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1, (25, counting_line), (1200, counting_line), (255,127,0), 3)
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= minimum_width) and (h >= minimum_height)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        center = paste_center(x, y, w, h)
        detec.append(center)
        cv2.circle(frame1, center, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(counting_line+offset) and y>(counting_line-offset):
                cars+=1
                cv2.line(frame1, (25, counting_line), (1200, counting_line), (0,127,255), 3)
                detec.remove((x,y))
                print("car is detected : "+str(cars))

    cv2.putText(frame1, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("Video Original" , frame1)
    cv2.imshow("Detect",dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
