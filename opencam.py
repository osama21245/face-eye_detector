import cv2 
import numpy as np

facedetect = cv2.CascadeClassifier("D:/Work/python apps/computerVision/haarcascade_frontalface_default.xml")
eyedetect = cv2.CascadeClassifier("D:/Work/python apps/computerVision/haarcascade_eye.xml")
smiledetect = cv2.CascadeClassifier("D:/Work/python apps/computerVision/haarcascade_smile.xml")

stream = cv2.VideoCapture(0)

while True :
    st,frame = stream.read()
    grayframe = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(grayframe,1.3,5)
    
    for (x,y,h,w) in faces :
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        faceonly = frame[x:x+w,y:y+h]
        eyes = eyedetect.detectMultiScale(faceonly,)
        
        
        for(xe,ye,he,we)in eyes :
            
            xeye = int((xe + (we/2))) -10
            yeye = int((ye + (he/2))) + 10
            cv2.putText(faceonly,"X",(xeye,yeye),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),5)
            
        
    
    
    cv2.imshow("facedetect",frame)
    cv2.waitKey(50)

  # Wait for any key press
cv2.destroyAllWindows()



