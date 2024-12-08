'''

Trains model
Note: Each time, we had a new set of weights, we manually updated it
So if you want to further train in the future, make sure you are using your most current weights

'''

from ultralytics import YOLO

#config_path = 'config.yaml'

# model = YOLOv10('/home/xavier/Robolab/runs/detect/train13/weights/last.pt')

#model.train(data=config_path, epochs = 100, batch = 16) 


model= YOLO('runs/detect/train14/weights/last.pt')
model.val( data='config.yaml')