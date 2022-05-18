import cv2
import numpy as np
import torch

from model import Net
from image import get_paper, putText

model = Net()
model.load_state_dict(torch.load('./checkpoint.pt'))

cap = cv2.VideoCapture(0)
while True:
   ret, frame = cap.read()

   crop = get_paper(frame)
   tensor = model.transforms(crop).unsqueeze(dim=0)
   number = torch.argmax(model(tensor))
   putText(frame, f'Number={number.item()}')

   cv2.imshow('frame', frame)
   cv2.imshow('crop', cv2.resize(crop, (128, 128)))
   key = cv2.waitKey(1)
   if  key == ord('q'):
      break
   elif key == ord('c'):
      cv2.imwrite('cap_frame.png', frame)
      cv2.imwrite('cap_crop.png', crop)

cap.release()
cv2.destroyAllWindows()
