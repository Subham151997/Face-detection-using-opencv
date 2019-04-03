
#Importing libraries
import cv2
import numpy as np

#Setting up the webcam
cap=cv2.VideoCapture(0)

#trained xml classifier for detecting face from lots of positive and negative image
face_cascade=cv2.CascadeClassifier("C:\\Users\\HP\\Anaconda3\\envs\\opencv-env\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml")

#Capturing frames from webcam and converting it to grayscale 
while(True):
    ret,frame=cap.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #detect face of different size
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=7)
    
    
    #setting our of interest 
    for(x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        
        img_item="myimage.png"
        cv2.imwrite(img_item,roi_gray)
        
        
        #setting a rectangle aroung our region of interest
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        
        
        
    #shwoing the frame    
    cv2.imshow('frame',frame)
    #Setting the condition of closing the window if esc key is press
    if cv2.waitKey(1)==27:
        break
#closing the webcam and destroying all windows    
cap.release()
cv2.destroyAllWindows()    


