# Import necessary modules and libraries
from flask import Flask, render_template, Response
import cv2
import time
import threading
import torch
import numpy as np
import detect

app = Flask(__name__)

frame_queue = []
thread_lock = threading.Lock()

# Refactor the core YOLOv5 detection logic from detect.py into a separate function



def process_frame():
    global frame_queue, thread_lock

    # Initialize the webcam capture
    cap = cv2.VideoCapture(0)
    count=0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
       

        frame = cv2.resize(frame, (640, 480))
        frame = detect.run(weights='NICbest.pt',source= '0', save_crop=True,conf_thres=0.7 )

        
       

       

        with thread_lock:
            if len(frame_queue) >= 5:
                frame_queue.pop(0)
            frame_queue.append(frame)

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

def generate_frames():
    global frame_queue, thread_lock

    while True:
        with thread_lock:
            if len(frame_queue) == 0:
                continue
            frame = frame_queue[0]

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index1():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_webcam')
def start_webcam():
    threading.Thread(target=process_frame).start()
    return 'Webcam started'

if __name__ == '__main__':
    app.run(debug=True)
    


