import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('model.pt')

# Print class names
print("Class Names:", model.names)

# Initialize camera
webcamera = cv2.VideoCapture(0)

if not webcamera.isOpened():
    print("Error: Unable to access webcam")
    exit()

while True:
    success, frame = webcamera.read()
    
    if not success:
        continue  # Skip if frame is not captured properly

    # Run YOLO detection on the frame
    results = model.track(frame, conf=0.5, imgsz=480)

    # Count detected objects
    total_objects = 0
    if results and results[0].boxes is not None:
        total_objects = len(results[0].boxes)

        print("\nDetected Objects:")  # Print header in terminal
        for i, (box, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.cls)):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers
            class_id = int(cls)  # Convert class index to integer
            class_name = model.names[class_id]  # Get class name
            
            # Calculate the center of the bounding box
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # Print detected object name & coordinates in terminal
            print(f"Object {i+1}: {class_name} - Center (X, Y) = ({center_x}, {center_y})")

            # Display class name and coordinates on the video frame
            label = f"{class_name} ({center_x}, {center_y})"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display total object count
    cv2.putText(frame, f"Total Objects: {total_objects}", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show detections
    cv2.imshow("Live Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
webcamera.release()
cv2.destroyAllWindows()
