#  Real-Time Object Detection using YOLOv8 and Raspberry Pi 4

This project demonstrates a complete object detection pipeline using the YOLOv8 model. The pipeline includes custom dataset preparation, training using Google Colab, and real-time deployment on a Raspberry Pi 4 for edge inference.


##  Project Workflow

1. **Dataset Preparation** – Labeled custom images using Roboflow.
2. **Model Training** – Trained YOLOv8 on Google Colab using the Ultralytics library.
3. **Model Deployment** – Deployed the trained `.pt` model on Raspberry Pi 4 for real-time detection using a USB camera.


##  Dataset Preparation

- The dataset was labeled and exported in YOLOv8 format using [Roboflow](https://roboflow.com).
- Download the dataset from:  
  ➤ **[👉 Dataset Download Link ([https://app.roboflow.com/...](https://app.roboflow.com/ds/dl8fgjgQwI?key=1pBOly0My3))**

**Labeling Tool Used:** Roboflow  
**Classes:** (e.g., apple, banana, stop sign, etc.)  
**Format:** YOLOv8 (TXT files with normalized coordinates)

---

## 🏋️ Model Training on Google Colab

Training was done using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) library in Google Colab.

### ✅ Requirements

```bash
!pip install ultralytics
```

### ✅ Train the Model

```python
from ultralytics import YOLO

# Load and train
model = YOLO("yolov8n.pt")  # Choose n/s/m/l/x based on your hardware
model.train(data="dataset.yaml", epochs=50, imgsz=640)
```

### ✅ Validate Performance

```python
metrics = model.val()
```

### ✅ Save and Export

```python
model.export(format="onnx")  # Optional: export to ONNX or TFLite
```
Trained model saved as: `best.pt`

---

##  Deployment on Raspberry Pi 4

### 📌 Setup

1. Flash Raspberry Pi OS on SD card and boot the Pi.
2. Connect a USB camera.
3. Install dependencies:

```bash
sudo apt update
sudo apt install python3-pip
pip3 install torch torchvision opencv-python ultralytics
```

###  Run the Model by using following detect.py file

```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("best.pt")

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 📊 Results

- **Model:** YOLOv8n
- **Epochs:** 50  
- **Accuracy:** ~XX% mAP@0.5 (insert actual results)
- **Speed:** Real-time on Raspberry Pi 4 with USB webcam

---

## Files and Structure

```
├── dataset/               # Roboflow Export (images + labels)
├── training_notebook.ipynb
├── best.pt                # Trained model
├── raspberry_pi_code.py   # Real-time inference script
├── README.md              # You're here!
```

---

##  Tools Used

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [Google Colab](https://colab.research.google.com/)
- [Raspberry Pi 4 Model B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)

---

## 📌 References

- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Labeling](https://docs.roboflow.com/)
- [Deploying on Pi](https://www.raspberrypi.com/documentation/)


