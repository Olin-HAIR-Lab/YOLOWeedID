'''
Uses trained yolo model to detect weeds and lettuce in real time on the real sense camera
'''
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLOv10 
import math

pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 1920,1080, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640,480, rs.format.z16, 30)

model = YOLOv10('runs/detect/train14/weights/last.pt')  # or another version of YOLOv10 (e.g., yolov10s.pt for small)

# object classes
classNames = ["Lettuce", "Weed"]

pipe.start(cfg)

while True:

    frame = pipe.wait_for_frames()
    # depth_frame = frame.get_depth_frame() (DONT NEED DEPTH)
    color_frame = frame.get_color_frame()

    results = model(color_frame, stream=True) # THIS SAYS THAT THE INPUT IS A STREAM

 

    # depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('rgb', color_image)
    # cv2.imshow('depth', depth_image)
    results = model(color_image, stream=True) # THIS SAYS THAT THE INPUT IS A STREAM

    
     # coordinates
    for r in results:
        boxes = r.boxes # gets all the boxes 

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(color_image, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', color_image)

    if cv2.waitKey(1) == ord('q'):
        break

pipe.stop()
cfg.release()
cv2.destroyAllWindows()