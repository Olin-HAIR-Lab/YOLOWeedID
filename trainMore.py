import os 

from ultralytics import YOLOv10 

config_path = 'config.yaml'

# model = YOLOv10('/home/xavier/Robolab/runs/detect/train13/weights/last.pt')

#model.train(data=config_path, epochs = 100, batch = 16) 


model= YOLOv10('runs/detect/train14/weights/last.pt')
model.val()