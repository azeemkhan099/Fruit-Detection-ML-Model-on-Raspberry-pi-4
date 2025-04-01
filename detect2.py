import torch
from ultralytics import YOLO
import cv2
import sys

# Load the trained model
model = YOLO("best.pt")  # Ensure the model file is in the same directory

# Function to run inference
def run_inference(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv8 inference
        results = model(frame)

        # Draw bounding boxes on the frame
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Ensure it's on CPU
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("YOLOv8 Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry point: Get source from command-line arguments
if __name__ == "__main__":
    if len(sys.argv) > 1:
        source = sys.argv[1]
        if source.isdigit():  # If it's a camera index (e.g., 0, 1)
            source = int(source)
        run_inference(source)
    else:
        print("Usage: python detect.py <source>")
        print("Example: python detect.py 0                 (for live camera)")
        print("Example: python detect.py video.mp4         (for video file)")
        print("Example: python detect.py image.jpg         (for an image — not supported in this loop)")
