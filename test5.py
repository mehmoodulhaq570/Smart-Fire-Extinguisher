# YOLOv8 detects objects in the webcam feed.
# MiDaS estimates depth for the entire frame.
# Each detected object gets a depth value based on its center.
# Overlay detection + depth info on screen.
# Using the depth too for the distance and angle
# Taking five readings and averaging them for better accuracy.

import cv2
import torch
import numpy as np
import time
import math
from ultralytics import YOLO
from torchvision.transforms import Compose, Normalize, ToTensor

# Load YOLO model
yolo_model = YOLO("best.pt")  # Replace with your YOLO model

# Load MiDaS for depth estimation
model_type = "DPT_Large"  # Options: DPT_Large, DPT_Hybrid, MiDaS_small
midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# Preprocessing function for MiDaS
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Open webcam
cap = cv2.VideoCapture(0)

# Get frame dimensions
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
origin_x, origin_y = frame_width / 2, frame_height / 2  # Image center

# Storage for readings
angle_readings = []
distance_readings = []
# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Adjusted focal length to minimize error based on table

def calculate_angle(x, y, depth):
    """ Compute angle θx between detected object and image center, incorporating depth """
    dx = x - origin_x
    dy = origin_y - y  # Invert Y since OpenCV origin is top-left
    
    # Compute horizontal angle
    theta_x = math.degrees(math.atan2(abs(dy), dx))

    if dx < 0:  
        theta_x = 180 - theta_x  # Adjust for left-side quadrant

    return theta_x

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1️⃣ Run YOLO object detection
    results = yolo_model(frame)
    
    # 2️⃣ Run MiDaS depth estimation
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (384, 384))
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_map = midas(img)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

    # Apply color map for visualization
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    # 3️⃣ Process detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0].item()  # Confidence score

            if conf > 0.5:  # Only consider confident detections
                bbox_center_x = (x1 + x2) // 2
                bbox_center_y = (y1 + y2) // 2
                bbox_width = x2 - x1
                
                estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width

                # Get depth value at object center
                object_depth = depth_map[bbox_center_y, bbox_center_x] / 255.0  # Normalize (0 to 1)
                real_distance = object_depth * (estimated_distance * 2200 * 1.45) + 5  # Scale (adjust based on camera calibration)

                # Compute corrected angles
                theta_x_degrees = calculate_angle(bbox_center_x, bbox_center_y, real_distance)

                # Store first 5 readings
                if len(distance_readings) < 5:
                    distance_readings.append(real_distance)
                    angle_readings.append(theta_x_degrees)
                
                # Draw bounding box and annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {theta_x_degrees:.2f}°", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 4️⃣ Display results
    combined = cv2.addWeighted(frame, 0.6, depth_colormap, 0.4, 0)
    cv2.imshow("YOLO + Depth", combined)

    # If 5 readings are stored, print them and wait 5 seconds
    if len(distance_readings) == 5:
        print("\n📌 Averaged Readings:")
        print(f"📏 Distance: {sum(distance_readings) / 5:.2f} cm")
        print(f"📐 Angle: {sum(angle_readings) / 5:.2f}°")
        
        distance_readings.clear()
        angle_readings.clear()
        
        time.sleep(5)  # Wait before next set of readings

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
