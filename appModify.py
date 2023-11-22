from flask import Flask, render_template, Response
import cv2
import time
import threading
import detect

app = Flask(__name__)

cap = cv2.VideoCapture(0)

frame_queue = []
thread_lock = threading.Lock()
pTime = 0

def process_frame():
    global frame_queue, thread_lock ,pTime

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        

        # frame = cv2.resize(frame, (640, 480))
        frame = detect.run(weights='NICbest.pt', source='0', save_crop=True , conf_thres=0.7)
        
        cTime = time.time()
        s = frame.shape
        # print(s)
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 0, 0), 2)

        with thread_lock:
            if len(frame_queue) >= 5:
                frame_queue.pop(0)
            frame_queue.append(frame)

        time.sleep(0.01)

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
