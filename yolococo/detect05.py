import cv2
import numpy as np
import torch
from yolov5 import detect  # Assuming you have YOLOv5 code in a folder named 'yolov5'

# Load YOLOv5 model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # You can use a different YOLOv5 model if needed

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt',force_reload=True)

# Load OpenPose model
net = cv2.dnn.readNetFromTensorflow('path/to/openpose/pose_iter_160000.caffemodel',
                                    'path/to/openpose/pose_deploy_linevec.prototxt')

# Open the video file
cap = cv2.VideoCapture('20231004_同學.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Perform YOLOv5 object detection on the frame
    results = model(frame)

    # Extract the detected people's bounding boxes
    person_boxes = results.pred[results.pred[:, 5] == 0][:, :4].cpu().numpy()

    # Perform multi-person pose detection on the detected people
    for box in person_boxes:
        x, y, w, h = box.astype(int)
        person_roi = frame[y:y + h, x:x + w]  # Extract the region of interest for the person

        # Resize the person_roi for OpenPose input (assuming 368x368 is the input size)
        person_roi_resized = cv2.resize(person_roi, (368, 368))

        # Normalize the input for OpenPose
        person_roi_normalized = person_roi_resized / 255.0

        # Perform forward pass with OpenPose
        blob = cv2.dnn.blobFromImage(person_roi_normalized, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True,
                                     crop=False)
        net.setInput(blob)
        output = net.forward()

        # Process the output to get pose keypoints
        # You'll need to define how to interpret the output based on OpenPose's output format

        # Draw the pose keypoints on the frame
        # You can use the cv2 functions to draw keypoints on the frame

    # Display or save the frame with keypoints
    cv2.imshow('Multi-Person Pose Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
