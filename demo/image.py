import cv2
import numpy as np

def get_paper(frame):
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

   low = np.array([0, 0, 200])
   upp = np.array([145, 50, 255])
   mask = cv2.inRange(hsv, low, upp)

   contours, hi = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   crop = np.zeros_like(frame)
   if len(contours) > 0:
      c = max(contours, key = cv2.contourArea)
      x,y,w,h = cv2.boundingRect(c)
      cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

      crop = frame[y:y+h, x:x+w]
   crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
   crop = cv2.resize(crop, (28, 28))
   crop = cv2.bitwise_not(crop)
   return crop

def putText(frame, text):
   cv2.putText(frame, text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 
                      3, (0, 0, 255), 8, cv2.LINE_AA)
