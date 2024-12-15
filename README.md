## Real-Time Face Detection with YOLO and DeepFace

This project is a real-time face detection application using YOLO for object detection and DeepFace for analyzing facial expressions and predicting age. The application can run on desktop or as a Flask-based web app for video streaming.

## Features
- **Human Detection**: Uses YOLO to detect objects, specifically humans (label 0 from the COCO dataset).
- **Facial Expression Prediction**: Analyzes detected human faces to determine their dominant facial expression using DeepFace.
- **Age Prediction**: Predicts the age of detected human faces with DeepFace.
- **Real-Time Video Streaming**: Streams annotated video in real-time via Flask and a web browser.

## Algorithms and Models Used
1. **YOLOv8 (Ultralytics)**: A state-of-the-art object detection model trained on the COCO dataset, used for detecting humans in video frames.
2. **DeepFace**: A library for facial analysis, used for predicting emotions and age.

## System Requirements
- Python 3.8+
- OpenCV for video processing
- Flask for server and video streaming
- YOLO (Ultralytics) for object detection
- DeepFace for facial analysis

## How to Use

### 1. Install Dependencies
Ensure Python is installed. Install the required dependencies by running:
```bash
pip install flask ultralytics deepface opencv-python torch torchvision
```

### 2. Download YOLO Model
Download a pretrained YOLO model (e.g., `yolov8n.pt`) from [Ultralytics](https://github.com/ultralytics/yolov8) and save it in the `models/` folder. Replace the model path in the code if necessary.

### 3. Run the Application
#### Desktop Mode
To run the desktop application with direct camera access, execute:
```bash
python app.py
```

#### Flask Mode
To run the Flask-based web app for real-time video streaming, move the Flask version of `app.py` from the `archive/` folder to the main directory and execute:
```bash
python app.py
```
Access the application in your browser at `http://127.0.0.1:5000/`.

## Features Explanation
- **Object Detection**: YOLO detects humans in the video frame and draws bounding boxes around them.
- **Facial Analysis**: For each detected human face, DeepFace analyzes:
  - **Dominant Emotion**: Identifies the most likely emotional expression (e.g., happy, sad, angry).
  - **Age**: Predicts the approximate age of the detected face.
- **Annotations**: Displays age and emotion information on the live video feed.

## Notes
- Ensure your camera is connected and accessible.
- For issues loading YOLO or DeepFace models, verify the dependencies and compatibility of your system.

## Troubleshooting
- **YOLO Model Issues**: Ensure the correct model file path is specified and the model is compatible with the Ultralytics library.
- **DeepFace Analysis Errors**: If faces are not detected, ensure the image cropping coordinates are correct and that `enforce_detection` is set appropriately.

For further assistance, consult the [TensorFlow Documentation](https://www.tensorflow.org/install) or [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/).

