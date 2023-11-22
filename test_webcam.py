import cv2

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    img = cv2.resize(img, (1280, 670))

    cv2.imshow('img',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

