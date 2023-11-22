import cv2
import torch
import numpy as np
import os

path = 'NICbest.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

cap = cv2.VideoCapture(0)
count = 0

output_folder1 = 'detected_imagesFront'
output_folder2 = 'detected_imagesBack'
os.makedirs(output_folder1, exist_ok=True)
os.makedirs(output_folder2, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    results = model(frame)
    frame_with_boxes = np.squeeze(results.render())

    for detection in results.pred[0]:
        class_id, confidence, *bbox = detection[:5]  # Fix unpacking here

        if int(class_id) == 0:  # Assuming class 0 represents the NIC card front
            x1, y1, x2, y2 = map(int, bbox)
            nic_card = frame[y1:y2, x1:x2]

            # Save the detected NIC card image
            image_name = os.path.join(output_folder1, f'nic_card_{count}.jpg')
            cv2.imwrite(image_name, nic_card)

        elif int(class_id) == 1:  # Assuming class 0 represents the NIC card back
            x1, y1, x2, y2 = map(int, bbox)
            nic_card = frame[y1:y2, x1:x2]

            # Save the detected NIC card image
            image_name = os.path.join(output_folder2, f'nic_card_{count}.jpg')
            cv2.imwrite(image_name, nic_card)

    cv2.imshow("FRAME", frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
