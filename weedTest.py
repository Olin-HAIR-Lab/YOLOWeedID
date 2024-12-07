from ultralytics import YOLO
import cv2
import math 
# start webcam
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

# USE THE YOLOTEST CONDA ENVIRONMENT

# model
model = YOLO('./runs/detect/train14/weights/last.pt')  # or another version of YOLOv10 (e.g., yolov10s.pt for small)

# object classes
classNames = ["Lettuce", "Weed"]


while True:
    success, img = cap.read()
    results = model(img, stream=True) # THIS SAYS THAT THE INPUT IS A STREAM

    # coordinates
    for r in results:
        boxes = r.boxes # gets all the boxes 

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

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

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'): # STOP CODE
        break

cap.release()
cv2.destroyAllWindows()