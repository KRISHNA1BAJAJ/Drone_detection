# Advanced-Aerial-Drone-Detection-System

This project demonstrates real-time drone detection using YOLOv5 and OpenCV. It detects drones in real-time and displays a warning when a drone is detected inside or near a defined rectangle.


## Requirements

- Python 3.x
- OpenCV
- PyTorch
- YOLOv5
- Numpy
- PIL

## Installation

1. Clone the repository:

   - git clone https://github.com/KRISHNA1BAJAJ/Drone_detection.git

2. Install the required Python libraries:

   - pip install opencv-python torch numpy pillow

3. Download the YOLOv5 model:

   - Visit the official YOLOv5 repository: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
   - Download the desired pre-trained model weights (e.g., `best.pt`) and place them in the project directory.

## Usage

1. Run the 'ddd.py' script:
   - python Advanced_Drone_Detection.py

2. The script will open a live video feed from the default camera.
   - To create a rectangle, click and drag the mouse on the video feed to define the four corners of the rectangle.
   - The rectangle can be adjusted by dragging the corners.
   - A warning message will be displayed whenever a drone is detected inside or near the rectangle.

3. Press 'q' to quit the program.


