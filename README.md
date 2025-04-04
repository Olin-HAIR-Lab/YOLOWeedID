# Weed Identification Using Yolo

By Xavier Nishikawa, Naomi Francis, Irene Hong, Ben Ricket

## Project Description

This repo uses YOLOV10 to identify weeds and lettuce using bounding boxes or semantic segmentation.

[![Weed Identification Demo](https://img.youtube.com/vi/ZdfIrKPBTg4/maxresdefault.jpg
)](https://youtu.be/ZdfIrKPBTg4)
## Install Instructions

Clone the repository in desired directory
```bash
https://github.com/Olin-HAIR-Lab/YOLOWeedID.git
```

This project uses Python 3.12.2

Download the following requirements once you have cloned the repo

```bash
pip install -r requirements.txt
```

## Methodology

Methodology
The YOLOWeedID project employs the YOLOv10 object detection model to distinguish between lettuce plants and weeds in agricultural environments. This methodology ensures precise identification, facilitating targeted weed management strategies.

Data Collection and Preparation

High-resolution images of lettuce fields are captured under diverse environmental conditions to encompass various growth stages and lighting scenarios. These images are annotated to label lettuce plants and different weed species, creating a comprehensive dataset for model training.

Data Augmentation

To enhance the model's robustness, data augmentation techniques are applied using the augmentImage.ipynb script. This process involves transformations such as rotations, scaling, and color adjustments, effectively increasing the dataset's variability.

Model Configuration

The YOLOv10 model is configured through the config.yaml file, which specifies parameters including input image size, batch size, learning rate, and the number of training epochs. These settings are optimized to balance detection accuracy and computational efficiency.

Training Process

Training is initiated with the train.py script, utilizing the prepared dataset and configuration settings. The model learns to identify and differentiate between lettuce plants and weeds by minimizing detection errors over multiple epochs.

Inference and Prediction

Post-training, the model's performance is evaluated using the Predict module. New images are processed to generate bounding boxes around detected lettuce and weed instances, enabling visual assessment of the model's accuracy.

Integration with Camera Systems

The CameraCode module facilitates real-time image acquisition from camera devices, allowing seamless integration of the trained model into field applications for live weed detection.

Conclusion

By leveraging YOLOv10's advanced real-time object detection capabilities, the YOLOWeedID project offers an efficient solution for distinguishing between crops and weeds. This approach contributes to sustainable and precise agricultural practices.

## Troubleshooting

If you have trouble running the code, contact Xavier Nishikawa 
