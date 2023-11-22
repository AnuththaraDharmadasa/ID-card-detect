import cv2
import torch
import numpy as np
from flask import Flask, render_template, Response


path ='NICbest.pt'

model = torch.hub.load('ultralytics/yolov5', 'custom',path, force_reload=True)

cap = cv2.VideoCapture(0)
count=0
while True:
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame,(1280,720))

    results = model(frame)
    frame = np.squeeze(results.render())
    cv2.imshow("FRAME",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
