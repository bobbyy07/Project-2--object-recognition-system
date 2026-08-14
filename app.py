from flask import Flask, Response, render_template
from ultralytics import YOLO
import cv2
import cvzone
import math
import os

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 480)

model = YOLO('yolo-weights/yolov8n.pt')

classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "dining table", "toilet", "TV monitor", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def generate_frames():
    while True:
        success, image = cap.read()
        results = model(image, conf=0.5, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(image, (x1, y1, w, h))

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Class Name
                cls = int(box.cls[0])
                cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(35, x1), max(35, y1)))
        # --- YOUR YOLO LOGIC ENDS HERE ---

        # Encode the frame into JPEG format
        ret, buffer = cv2.imencode('.jpg', image)
        key = cv2.waitKey(3)
        if key & 0xFF == ord('k'):
            break
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
