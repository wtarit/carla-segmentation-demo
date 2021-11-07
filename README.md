# Carla Segmentation Demo  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/Hoy_W717_Bo/0.jpg)](https://youtu.be/Hoy_W717_Bo)

Image segmentation code is based on https://github.com/qubvel/segmentation_models.pytorch 

## Dataset
[**Download Dataset**](https://github.com/wtarit/carla-segmentation-demo/releases/download/v0.0.1/Processed_road_dataset.zip)  
Dataset ถูกเก็บบน map Town07 ช่วงถนน 2 lane  
Image size 640x480  
Dataset เก็บ label ไว้ใน Red color channal  
Label    | Class
---------|--------
0        | Background
1        | Background

RGB image             |  Processed Segmentation Label Mask Visualized
:-------------------------:|:-------------------------:
![RGB image](https://raw.githubusercontent.com/wtarit/carla-segmentation-demo/main/img/rgb.png)  |  ![Processed Segmentation Label Mask Visualized](https://raw.githubusercontent.com/wtarit/carla-segmentation-demo/main/img/mask_vis.png)

## Model
**Trained Model**
Architectures | Encoders | Weight
--------------|-----------|----------
FPN | timm-mobilenetv3_large_075 | [Download](https://github.com/wtarit/carla-segmentation-demo/releases/download/v0.0.1/FPN_timm-mobilenetv3_large_075.pth)
Unet | timm-mobilenetv3_large_075 | [Download](https://github.com/wtarit/carla-segmentation-demo/releases/download/v0.0.1/Unet_timm-mobilenetv3_large_075.pth)
