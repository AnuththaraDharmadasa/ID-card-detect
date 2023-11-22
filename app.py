from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import torch
import detect  # Import your YOLOv5 detection code

app = Flask(__name__)
socketio = SocketIO(app)

cap = cv2.VideoCapture(0)  # Open default camera (you may need to change this)

@app.route('/')
def index():
    return render_template('start.html')

def generate():
    while True:
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@socketio.on('start_stream')
def handle_start_stream():
    socketio.emit('update_status', {'status': 'Streaming Started'})
    socketio.emit('video_feed', {'source': '/video_feed'})

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('detect_objects')
def handle_detect_objects():
    result = run_yolo_detection()  # Modify this based on your YOLOv5 code
    socketio.emit('object_detection_result', {'result': result})

if __name__ == '__main__':
    socketio.run(app, debug=True)
