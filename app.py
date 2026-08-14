from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)

# ---------------------------------------------------
# Load YOLO model
# ---------------------------------------------------

MODEL_PATH = "yolov8n.onnx"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' was not found."
    )

model = YOLO(MODEL_PATH)

print("YOLO model loaded successfully!")


# ---------------------------------------------------
# Home page
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------
# Object detection
# ---------------------------------------------------

@app.route("/detect", methods=["POST"])
def detect():

    try:
        # Check if image was received
        if "image" not in request.files:
            return jsonify({
                "error": "No image received"
            }), 400

        file = request.files["image"]

        # Read image bytes
        image_bytes = file.read()

        # Convert bytes to numpy array
        np_array = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                "error": "Could not decode image"
            }), 400

        # ---------------------------------------------------
        # Run YOLO detection
        # ---------------------------------------------------

        results = model.predict(
            source=image,
            conf=0.5,
            verbose=False
        )

        # ---------------------------------------------------
        # Draw detections
        # ---------------------------------------------------

        for result in results:

            boxes = result.boxes

            for box in boxes:

                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                # Confidence
                confidence = float(box.conf[0])

                # Class ID
                class_id = int(box.cls[0])

                # Get class name
                class_name = model.names[class_id]

                # Label
                label = f"{class_name} {confidence:.2f}"

                # Draw rectangle
                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Text background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1
                )

                # Label text
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )

        # ---------------------------------------------------
        # Encode image as JPEG
        # ---------------------------------------------------

        success, buffer = cv2.imencode(".jpg", image)

        if not success:
            return jsonify({
                "error": "Could not encode image"
            }), 500

        return Response(
            buffer.tobytes(),
            mimetype="image/jpeg"
        )

    except Exception as e:

        print("Detection error:", e)

        return jsonify({
            "error": str(e)
        }), 500


# ---------------------------------------------------
# Run locally
# ---------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
