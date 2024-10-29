from FunctionLibrary import *
import cv2
import time
import imutils
import matplotlib.pyplot as plt
import numpy as np

import firebase_admin
from firebase_admin import db
import json
from datetime import date



cred_obj = firebase_admin.credentials.Certificate('project-vehicle-tracking-64d90-firebase-adminsdk-pzj7n-0c08fc3bc3.json')
default_app = firebase_admin.initialize_app(cred_obj, {
	'databaseURL':'https://project-vehicle-tracking-64d90-default-rtdb.firebaseio.com/'
	})

ref = db.reference("/days")


# ref = db.reference("/Books/Best_Sellers")
# import json
# with open("books.json", "r") as f:
#	file_contents = json.load(f)

#for key, value in file_contents.items():
#	ref.push().set(value)

tracker=EuclideanDistTracker()
PTime=0
obj_det=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=40)

counting_line = 550
cars= 0
offset=6 # Error allowed between pixel
delay= 60 # Video FPS
minimum_width=80 # Minimum width of the rectangle
minimum_height=80 # Minimum height of the rectangle

def paste_center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy


WebcamIsUsing = False


if WebcamIsUsing:
    cap=cv2.VideoCapture(1)
else:
    cap=cv2.VideoCapture("video.mp4")

subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    _,img=cap.read()
    if WebcamIsUsing:
        img=imutils.resize(img,width=1200)
    h,w,_,=img.shape
    # roi=img[340: 720,500: 800]
    roi=img[100: 720,100: 1500]
    mask=obj_det.apply(roi)
    _,mask=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    cont,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    det=[]
    detec = []

    ##############
    tempo = float(1/delay)
    time.sleep(tempo)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(9,9),5)
    img_sub = subtracao.apply(blur)
    # wide open
    dilat = cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx (dilat, cv2. MORPH_CLOSE , kernel)
    dilatada = cv2.morphologyEx (dilatada, cv2. MORPH_CLOSE , kernel)
    contorno,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(img, (25, counting_line), (1200, counting_line), (255,127,0), 3)
    for(i,c) in enumerate(contorno):
        (x,y,w,h) = cv2.boundingRect(c)
        validar_contorno = (w >= minimum_width) and (h >= minimum_height)
        if not validar_contorno:
            continue

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        center = paste_center(x, y, w, h)
        detec.append(center)
        cv2.circle(img, center, 4, (0, 0,255), -1)

        for (x,y) in detec:
            if y<(counting_line+offset) and y>(counting_line-offset):
                cars+=1
                cv2.line(img, (25, counting_line), (1200, counting_line), (0,127,255), 3)
                detec.remove((x,y))
                print("car is detected : "+str(cars))
    ##############

    cv2.line(img, (25, counting_line), (1200, counting_line), (255,127,0), 3)
    for cnt in cont:
        area=cv2.contourArea(cnt)
        if area>100:
            #cv2.drawContours(roi,[cnt],-1,(0,255,0),2)
            x,y,w,h=cv2.boundingRect(cnt)
            det.append([x,y,w,h])


    CTime=time.time()
    fps=1/(CTime-PTime)
    PTime=CTime

    x_line = []
    y_line = []
    w_line = []
    h_line = []

    boxes_ids=tracker.update(det)
    for box in boxes_ids:
        x,y,w,h,id=box
        SpeedEstimatorTool=SpeedEstimator([x,y],fps)
        speed=SpeedEstimatorTool.estimateSpeed()
        cv2.putText(roi,str(id)+": "+str(speed)+"Km/h",(x,y-15),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),3)
        x_line.append(x)
        y_line.append(y)
        w_line.append(w)
        h_line.append(h)

    cv2.putText(img, "VEHICLE COUNT : "+str(cars), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),5)
    cv2.imshow("mask",mask)
    cv2.imshow("roi",roi)
    cv2.imshow("img",img)

    x_new = np.array(x_line)
    y_new = np.array(y_line)
    w_new = np.array(w_line)
    h_new = np.array(h_line)



    key=cv2.waitKey(30)
    if key==113: #113=Q
        ref.push().set(json.dumps({"date": str(date.today()), "Number of Vehicles": str(cars)}))
        plt.scatter(x_new,y_new)
        plt.scatter(w_new,h_new)
        plt.show()
        break
