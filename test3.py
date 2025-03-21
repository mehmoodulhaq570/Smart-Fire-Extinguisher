# I have tried to adjsut the distance formula and angle
# Angle is not moving to 180 degree (Fix it)

import cv2
import math
import serial  # For Arduino communication
from ultralytics import YOLO

# Load YOLOv11 model
model = YOLO("best.pt")

# Open webcam (use 'video.mp4' for video input)
cap = cv2.VideoCapture(0)  

# Get frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
origin_x, origin_y = frame_width / 2, frame_height / 2  # Image center

# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Adjusted focal length to minimize error based on table

def calculate_angle(x, y):
    """ Compute angle θx between detected object and image center """
    dx = x - origin_x
    dy = origin_y - y  # Invert Y since OpenCV's origin is top-left
    
    theta_x = math.degrees(math.atan2(abs(dy), dx))  # Use abs() to avoid negative angles
    
    if dx < 0:  
        theta_x = 180 - theta_x  # Adjust for left-side quadrant

    return theta_x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score

            if conf > 0.5:  # Process only confident detections
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                bbox_width = x2 - x1

                # Compute corrected angles
                theta_x_degrees = calculate_angle(bbox_center_x, bbox_center_y)

                # Estimate distance with corrected formula
                estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                real_distance = (estimated_distance * 2200 * 1.45) + 5  # Adjusted scale factor

                # Draw detection & annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {theta_x_degrees:.2f}°", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Print debug info
                print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                print(f"Corrected Angle: {theta_x_degrees:.2f}°")
                print(f"Estimated Distance: {real_distance:.2f} cm")

    # Display frame
    cv2.imshow("YOLOv11 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
