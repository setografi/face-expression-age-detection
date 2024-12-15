# Menggunakan Flask dengan konsep Streaming Video 

from flask import Flask, render_template, Response
from ultralytics import YOLO
from deepface import DeepFace
import cv2
import numpy as np

app = Flask(__name__)

# Load YOLO Model (pretrained)
yolo_model = YOLO("models/yolov8n.pt")  # YOLO Nano model (COCO pretrained)

# Fungsi untuk memproses frame
def process_frame(frame):
    try:
        # Deteksi objek dengan YOLO
        results = yolo_model(frame)

        # Hasil anotasi pada frame
        annotated_frame = results[0].plot()
        objects = []

        for result in results[0].boxes:
            box = result.xyxy.tolist()[0]  # Ambil box pertama
            label = result.cls[0]  # Ambil label pertama

            # Jika objek adalah manusia, prediksi ekspresi dan umur
            if int(label) == 0:  # Label 0 = "person" di YOLO COCO
                x1, y1, x2, y2 = map(int, box)
                cropped_face = frame[y1:y2, x1:x2]
                
                try:
                    # Prediksi ekspresi dan umur menggunakan DeepFace
                    analysis = DeepFace.analyze(cropped_face, actions=['emotion', 'age'], enforce_detection=False)
                    objects.append({
                        "label": "person",
                        "age": analysis[0].get('age'),
                        "emotion": analysis[0].get('dominant_emotion'),
                        "coordinates": box
                    })
                except Exception as e:
                    print(f"Error analyzing face: {e}")

        return annotated_frame, objects
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame, []

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            annotated_frame, objects = process_frame(frame)
            
            # Tambahkan informasi ke frame
            for obj in objects:
                cv2.putText(annotated_frame, 
                            f"Age: {obj['age']}, Emotion: {obj['emotion']}", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 255, 0), 
                            2)

            # Encode frame untuk streaming
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)