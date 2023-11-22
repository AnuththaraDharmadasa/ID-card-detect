import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

imgPath = "R1.jpeg"

reader = easyocr.Reader(['en'], gpu =False)
result = reader.readtext(imgPath)
result

top_left =tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_COMPLEX_SMALL


img = cv2.imread(imgPath)
# spacer = 100
# for detection in result:
     
#     top_left = tuple(detection[0][0])
#     bottom_right = tuple(detection[0][2])
#     text = detection[1]
#     img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
#     img = cv2.putText(img,text,top_left,font,3,(0,0,0),1,cv2.LINE_AA)
# #     img = cv2.putText(img,text,(20,spacer), font, 2,(0,0,0),2,cv2.LINE_AA)
#     spacer+=15
cv2.imshow('img',img)
# plt.imshow(img)
# plt.show