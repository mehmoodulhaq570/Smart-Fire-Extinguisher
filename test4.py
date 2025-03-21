# Store the last five readings of distance and angle.
# Calculate the average once five readings are collected.
# Display and print the results.
# Wait for 5 seconds before collecting new readings.

import cv2
import math
import time  # For delay
from collections import deque  # To store last five readings
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)  

# Get frame size
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
origin_x, origin_y = frame_width / 2, frame_height / 2  # Image center

# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Adjusted focal length to minimize error

# Store last 5 readings of distance & angle
distance_readings = deque(maxlen=5)
angle_readings = deque(maxlen=5)

# Timing variables
display_start_time = None  # Track display time
pause_start_time = None  # Track pause time
display_duration = 4  # Seconds to show average readings
pause_duration = 5  # Seconds to wait before collecting new readings
display_active = False  # Flag to track display state

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

    detected = False  # Track if an object is detected

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0].item()  # Confidence score

            if conf > 0.5:  # Process only confident detections
                detected = True
                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                bbox_width = x2 - x1

                # Compute corrected angles
                theta_x_degrees = calculate_angle(bbox_center_x, bbox_center_y)

                # Estimate distance with corrected formula
                estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                real_distance = (estimated_distance * 2200 * 1.45) + 5  # Adjusted scale factor

                # Store readings
                distance_readings.append(real_distance)
                angle_readings.append(theta_x_degrees)

                # Draw detection & annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)

    current_time = time.time()

    if detected and len(distance_readings) == 5 and not display_active:
        # Calculate averages
        avg_distance = sum(distance_readings) / len(distance_readings)
        avg_angle = sum(angle_readings) / len(angle_readings)

        # Start display timer
        display_start_time = current_time
        display_active = True

        # Print to console
        print(f"Average Distance: {avg_distance:.2f} cm")
        print(f"Average Angle: {avg_angle:.2f}°")

    # Display results for 4 seconds
    if display_active:
        cv2.putText(frame, f"Avg Distance: {avg_distance:.2f} cm", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        cv2.putText(frame, f"Avg Angle: {avg_angle:.2f}°", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if current_time - display_start_time > display_duration:
            # Clear readings and start pause
            distance_readings.clear()
            angle_readings.clear()
            display_active = False
            pause_start_time = current_time  # Start 5-sec pause

    # Pause before collecting new readings
    if pause_start_time and current_time - pause_start_time < pause_duration:
        pass  # Do nothing (pause active)

    # Display frame
    cv2.imshow("YOLO Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
