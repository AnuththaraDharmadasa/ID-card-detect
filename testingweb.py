from flask import Flask, render_template, Response
import cv2
import detect
  # Import your detection module

app = Flask(__name__)

camera = cv2.VideoCapture(0)
Detector = detect  # Webcam feed


def generate_frames():
    while True:
        success, frame = camera.read()
        
        if not success:
            break
        else:

            frame = Detector.run(weights='NICbest.pt', source='0', save_crop=True , conf_thres=0.7)
        
            
            success, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('start.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)
# @app.route('/detect_objects', methods=['POST'])
# def detect_objects():
#     # Perform object detection
#     frame = cv2.resize(frame, (1280, 720))
#     frame=Detector.findPose(frame)
#     count, total_dura = detector1.Count(frame)
    

#     # Process detections and prepare them for display
#     # ... (convert to JSON, overlay boxes on images, etc.)

#     return render_template('start.html',frame=frame)
    # return detections