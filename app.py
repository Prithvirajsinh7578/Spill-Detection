from flask import Flask, render_template, request, Response
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("best.pt")

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ---------------- IMAGE UPLOAD PROCESSING --------------------
@app.route("/", methods=["GET", "POST"])
def home():
    output_image = None

    if request.method == "POST":
        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]
        img_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
        file.save(img_path)

        img = cv2.imread(img_path)

        # Faster + cleaner inference
        results = model(img)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            label = f"Spill {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")
        cv2.imwrite(output_path, img)
        output_image = "output.jpg"

    return render_template("index.html", output_image=output_image)


# ---------------- LIVE STREAM DETECTION ---------------------
cap = cv2.VideoCapture(0)  # keep camera open globally


def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Much faster than model.predict()
        results = model.track(frame, persist=True)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            score = float(box.conf[0])
            label = f"Spill {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_data = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_data + b"\r\n"
        )


@app.route("/live")
def live():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
