from re import L
import cv2
import numpy as np
import time

cam=cv2.VideoCapture(0)
prev= None
dist = lambda x1, y1, x2, y2: (x1-x2)**2+(y1-y2)**2

while True:
    ret, frame=cam.read()
    if not ret: break


    grayFrame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurFrame=cv2.GaussianBlur(grayFrame, (15, 15), 0)

    hsv_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # low=np.array([5,138,115]) #orange ball
    # high=np.array([161,244,255])
    low=np.array([33,136,57])
    high=np.array([81,232,228])
    ball_mask=cv2.inRange(hsv_frame, low, high)
    circles=cv2.HoughCircles(blurFrame, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=45, minRadius=40, maxRadius=300)
    contours, hierarchy = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if circles is not None and len(contours)!=0: 
        circles=np.uint16(np.around(circles))
        chosen=None
        for i in circles[0, :]:
            if chosen is None: chosen=i
            if prev is not None: 
                if dist(chosen[0], chosen[1], prev[0], prev[1])<=dist(i[0], i[1], prev[0], prev[1]): 
                    chosen=i
                    cv2.circle(frame, (chosen[0], chosen[1]), chosen[2], (255,0,255),3)
        prev=chosen
        for c in contours:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
                    
        

    # color=cv2.bitwise_and(frame, frame, mask=ball_mask)
    cv2.imshow('frame',frame)
    # cv2.imshow('mask', color)

    if cv2.waitKey(1)==ord('x'):
        break
cam.release()
cv2.destroyAllWindows()
