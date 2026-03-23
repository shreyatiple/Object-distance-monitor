#pip install ultralytics opencv-python
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (you can train custom for book & bottle for better accuracy)
model = YOLO("yolov8n.pt")

# Open webcam/video
cap = cv2.VideoCapture(0)

# Calibration factor (IMPORTANT)
# You need to adjust this based on real measurement
PIXEL_TO_CM = 0.026  # example value (change after calibration)

def get_distance(box1, box2):
    # box format: (x1, y1, x2, y2)

    # Bottom of book
    y_book_bottom = box1[3]

    # Top of bottle
    y_bottle_top = box2[1]

    pixel_distance = y_bottle_top - y_book_bottom

    if pixel_distance < 0:
        return 0  # touching

    return pixel_distance * PIXEL_TO_CM


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    book_box = None
    bottle_box = None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label == "book":
                book_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Book", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            elif label == "bottle":
                bottle_box = (x1, y1, x2, y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
                cv2.putText(frame, "Bottle", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Calculate distance
    if book_box and bottle_box:
        distance = get_distance(book_box, bottle_box)

        if distance <= 0.5:
            status = "NO GAP"
        else:
            status = "Gap"

        cv2.putText(frame, f"Distance: {distance:.2f} cm",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.putText(frame, status,
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Gap Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()