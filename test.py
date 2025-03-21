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

# # Arduino Serial Communication (Adjust COM port as needed)
# try:
#     ser = serial.Serial("COM3", 9600, timeout=1)  # Change "COM3" as per your Arduino port
#     print("Connected to Arduino")
# except Exception as e:
#     print(f"Error connecting to Arduino: {e}")
#     ser = None  # Continue execution without Arduino

# Object dimension parameters for distance estimation
KNOWN_WIDTH = 1.0  # Replace with actual object width in meters
FOCAL_LENGTH = 1.0  # Replace with actual focal length

def calculate_angle(x, y):
    """ Compute angle θx between detected object and image center """
    dx = x - origin_x
    dy = origin_y - y  # Invert Y since OpenCV's origin is top-left
    theta_x = math.atan2(dy, dx)
    theta_x_degrees = math.degrees(theta_x)

    # Normalize angle (-90 to +90 degrees → 0 to 180 for servo)
    servo_angle = int((theta_x_degrees + 90) * (180 / 180))
    
    return theta_x_degrees, servo_angle

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
                bbox_height = y2 - y1

                # Compute angles
                theta_x_degrees, servo_angle = calculate_angle(bbox_center_x, bbox_center_y)

                # # Send angle to Arduino (if connected)
                # if ser:
                #     try:
                #         ser.write(f"{theta_x_degrees}\n".encode())
                #         print(f"Servo angle sent: {theta_x_degrees:.2f}°")
                #     except Exception as e:
                #         print(f"Error sending data to Arduino: {e}")

                # Estimate distance using the bounding box width
                estimated_distance = (KNOWN_WIDTH * FOCAL_LENGTH) / bbox_width
                real_distance = (estimated_distance * 2300 * 1.5) + 10  # Scale factor

                # Draw detection & annotations
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(bbox_center_x), int(bbox_center_y)), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Angle: {theta_x_degrees:.2f}°", (x1, y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"Distance: {real_distance:.2f} cm", (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Print debug info
                print(f"Detection Center: ({bbox_center_x:.2f}, {bbox_center_y:.2f})")
                print(f"Servo Angle: {servo_angle}°")
                print(f"Estimated Distance: {real_distance:.2f} cm")

    # Display frame
    cv2.imshow("YOLOv11 Object Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# if ser:
#     ser.close()  # Close serial communication
