import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time

# Load the custom YOLO model
model = YOLO('last.pt', task='segment')

# Video paths
input_video_path = 'videos/video3.mp4'
output_video_path = 'videos/output.mp4'

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

# Process frames
frame_count = 0
start_time = time.time()

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break
        
    # Apply custom YOLO model for segmentation
    results = model(frame)[0]
    
    # Create a copy of the frame for overlay
    frame_with_mask = frame.copy()
    
    # Process segmentation results
    if results.masks is not None:  # Check if masks exist
        for mask, conf, cls in zip(results.masks.data, results.boxes.conf, results.boxes.cls):
            # Convert class index to integer
            cls = int(cls)
            
            # Only process confident detections
            if conf >= 0.3:
                # Handle mask processing based on dimensions
                if len(mask.shape) == 2:
                    seg_mask = (mask.byte().cpu().numpy() > 0.5).astype(np.uint8)
                else:
                    seg_mask = (mask.squeeze().byte().cpu().numpy() > 0.5).astype(np.uint8)
                
                # Resize mask to match frame dimensions
                seg_mask = cv2.resize(seg_mask, (frame_width, frame_height), 
                                   interpolation=cv2.INTER_NEAREST)
                
                # Create colored overlay based on class
                colored_mask = np.zeros_like(frame)
                if cls == 0:  # weed
                    colored_mask[:, :, 0] = seg_mask * 255  # Red for weeds
                else:  # lettuce
                    colored_mask[:, :, 1] = seg_mask * 255  # Green for lettuce
                
                # Apply the colored mask
                frame_with_mask = cv2.addWeighted(frame_with_mask, 0.99, colored_mask, 0.5, 0)
                
                # Find contours for the mask
                contours = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw contours and add labels
                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Get the centroid of the contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        # Add label near the centroid
                        label = f'{class_dict[cls]} {conf:.2f}'
                        cv2.putText(frame_with_mask, label, (cx, cy),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Write the processed frame
    out_video.write(frame_with_mask)
    
    # Update progress
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed_time = time.time() - start_time
        fps_current = frame_count / elapsed_time
        print(f'Processed frame {frame_count}/{total_frames} - Current FPS: {fps_current:.2f}')

# Calculate and print final statistics
end_time = time.time()
total_time = end_time - start_time
print(f"Total processing time: {total_time:.2f} seconds")
print(f"Average FPS: {total_frames / total_time:.2f}")

# Cleanup
video_capture.release()
out_video.release()
cv2.destroyAllWindows()
print(f'Output video saved to {output_video_path}')