import cv2
from ultralytics import YOLO
from deepface import DeepFace
import numpy as np

# Load YOLO Model (pretrained)
yolo_model = YOLO("models/yolov8n.pt")  # YOLO Nano model (COCO pretrained)

def process_frame(frame):
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
                
                # Tambahkan teks ke frame
                cv2.putText(annotated_frame, 
                            f"Age: {analysis[0].get('age')} Emotion: {analysis[0].get('dominant_emotion')}", 
                            (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, 
                            (0, 255, 0), 
                            2)
                
                objects.append({
                    "label": "person",
                    "age": analysis[0].get('age'),
                    "emotion": analysis[0].get('dominant_emotion'),
                    "coordinates": box
                })
            except Exception as e:
                print(f"Error analyzing face: {e}")

    return annotated_frame, objects

def main():
    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Baca frame dari kamera
        ret, frame = cap.read()
        
        if not ret:
            print("Gagal membaca frame dari kamera")
            break

        # Proses frame
        annotated_frame, objects = process_frame(frame)

        # Tampilkan frame yang sudah diproses
        cv2.imshow("Real-Time Detection", annotated_frame)

        # Cetak informasi objek yang terdeteksi
        if objects:
            print("Objek Terdeteksi:")
            for obj in objects:
                print(f"- Person: Umur {obj['age']}, Emosi {obj['emotion']}")

        # Keluar jika tombol 'q' ditekan
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Tutup kamera dan jendela
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()