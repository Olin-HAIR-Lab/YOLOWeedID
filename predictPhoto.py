from ultralytics import YOLOv10
import cv2
import numpy as np
dict = {0: 'weed', 1:'lettuce'}

def draw_boxes(image, results, conf_threshold=0.25):
    img = image.copy()
    
    boxes = results[0].boxes
    for box in boxes:
    # Get box coordinates, confidence and class
        coords = box.xyxy[0].cpu().numpy()
        conf = float(box.conf)
        cls = int(box.cls)
            
        if conf > conf_threshold:
            x1, y1, x2, y2 = map(int, coords)
                
                # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
            label = f'Class {dict[cls]}: {conf:.2f}'
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
    return img

# Load model
model = YOLOv10('yolov10/runs/detect/train3/weights/last.pt')

# Read image
image_path = 'predictPhotoData/0Cw6Uz8z.jpg'
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Draw boxes and save
output_image = draw_boxes(image, results)

# Save the result
output_path = 'predictPhotoData/output_detailed.jpg'
cv2.imwrite(output_path, output_image)

print(f"Annotated image saved to: {output_path}")

