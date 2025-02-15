import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model
# Update this path to point to your .pt file
model_path = r'C:\Users\krish\Desktop\Capstone\code3\ddd2\best.pt'  # or whatever your .pt file is named
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Set video source (webcam or video file)
cap = cv2.VideoCapture(0)

# Define the classes you want to detect
classes = ['Drone']

# Initialize the rectangle coordinates
rectangle_coords = [(50, 50), (250, 50), (250, 250), (50, 250)]
rectangle_drag = False
drag_corner = -1

def mouse_event(event, x, y, flags, param):
    global rectangle_coords, rectangle_drag, drag_corner
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, corner in enumerate(rectangle_coords):
            if abs(corner[0] - x) <= 10 and abs(corner[1] - y) <= 10:
                rectangle_drag = True
                drag_corner = i
                break
    
    elif event == cv2.EVENT_LBUTTONUP:
        rectangle_drag = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle_drag:
            rectangle_coords[drag_corner] = (x, y)

# Create window and set mouse callback
cv2.namedWindow('Drone Detection')
cv2.setMouseCallback('Drone Detection', mouse_event)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Run inference
        results = model(img, size=640)

        # Process detections
        for result in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = result.tolist()
            if conf > 0.5 and int(cls) < len(classes):  # Added class index check
                # Convert coordinates to integers
                bbox = tuple(map(int, (x1, y1, x2, y2)))
                
                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                
                # Add confidence score
                conf_text = f"{conf:.2%}"
                cv2.putText(frame, conf_text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Add coordinates
                coord_text = f"({(bbox[0] + bbox[2])//2}, {bbox[3]})"
                cv2.putText(frame, coord_text, (bbox[0], bbox[3] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check intersection with restricted area
                box_points = [(bbox[0], bbox[1]), (bbox[2], bbox[1]),
                            (bbox[2], bbox[3]), (bbox[0], bbox[3])]
                
                # Create numpy arrays for polygon intersection test
                box = np.array(box_points)
                restricted_area = np.array(rectangle_coords)
                
                # Check if any point of the detection box is inside the restricted area
                if any(cv2.pointPolygonTest(restricted_area, (x, y), False) >= 0 
                      for x, y in box_points):
                    cv2.putText(frame, "WARNING: Drone in Restricted Area!",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                              (0, 0, 255), 2)

        # Draw restricted area
        for i in range(4):
            # Draw corner points
            cv2.circle(frame, rectangle_coords[i], 5, (0, 255, 0), -1)
            # Draw lines between points
            next_point = rectangle_coords[(i + 1) % 4]
            cv2.line(frame, rectangle_coords[i], next_point, (0, 255, 0), 2)

        # Show frame
        cv2.imshow('Drone Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {str(e)}")

finally:
    cap.release()
    cv2.destroyAllWindows()