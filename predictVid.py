import cv2
import torch
from ultralytics import YOLO
import time

# Load the model
model = YOLO('/home/irene/Robolab/last.pt', task='segment')

# Video paths
input_video_path = '/home/irene/Robolab/videos/video3.mp4'
output_video_path = '/home/irene/Robolab/videos/output.mp4'

# Open the video
video_capture = cv2.VideoCapture(input_video_path)

# Get video properties
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_capture.get(cv2.CAP_PROP_FPS))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Class dictionary
class_dict = {0: 'weed', 1: 'lettuce'}
array = ['weed','lettuce']
# Process frames
frame_count = 0
startTime= time.time()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Apply YOLOv10 object detection
    results = model(frame)[0]
    # Process detections
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = result[:6]
        
        # Convert coordinates and class to proper types
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls = int(cls)  # Ensure class index is integer
        
        # Only process confident detections
        if conf >= 0.3:
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            
            # Prepare label text with class name and confidence
            label = f'{class_dict[cls]} {conf:.2f}'
            
            # Add label text
            cv2.putText(frame, 
                       label, 
                       (x1, max(y1 - 10, 20)),  # Ensure text doesn't go above frame
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, 
                       (255, 0, 0), 
                       2)

    # Write the processed frame
    out_video.write(frame)
    
    # Update progress
    frame_count += 1
    if frame_count % 10 == 0:  # Print every 10 frames to reduce console output
        print(f'Processed frame {frame_count}/{total_frames}')
endTime = time.time()

total = endTime - startTime
print("It took ", total, " seconds to run this.")
print("It was ", total_frames / total , 'fps (total frames/total seconds)')
# Cleanup
video_capture.release()
out_video.release()
cv2.destroyAllWindows()

print(f'Output video saved to {output_video_path}')