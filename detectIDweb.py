from flask import Flask, request, render_template
import detect
import cv2

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        detections = detect.run(weights='NICbest.pt', source='0', save_crop='results')
        return render_template('indexTest.html', detections=detections)
    return render_template('indexTest.html')

if __name__ == '__main__':
    app.run(debug=True)