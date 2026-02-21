# Road-Mapping-Scementic-Segmentation
# Scientific_Colloquium_Road_Mapping_Using_Sementic_Segmentation

### Table Of Contents
---
- [Overview](#overview)
- [Research Question](#research-question)

### Overview
---
Road perception is essential for intelligent vehicles. However, HD-map based and multi-sensor approaches are expensive and difficult to maintain in small-scale environments.

This project explores a camera-only road mapping pipeline using semantic segmentation to extract lane geometry from RGB images captured by an Intel RealSense camera.

Instead of relying on BEV-based complex systems like LaneSegNet, we use a pixel-wise segmentation model (DeepLabV3+) to directly detect drivable lane areas and extract geometry from segmentation masks.

### Research Question

“How well can semantic segmentation be applied to RGB images from Intel RealSense camera to extract lane geometry in a model city environment?”


### Hyperparameters
---
| Parameter     | Value     | 
|-----------|-------------|
| `Model` |DeepLabV3+|
| `Encode` |ResNet-50(Imagenet Pretrained)|
| `Framework` |Pytorch + segmentation_models_pytorch|
| `Input Size` |512 * 512|
| `Classes` |1 (lane)|
| `Batch Size` |4|
| `Epochs` |20|
| `Optimizer` |Adam|
| `Learning Rate` |1e-4|
| `Loss` |CrossEntropyLoss|
| `Train Loader` |shuffle=True, workers=2|
| `Val Loader` |shuffle=False, workers=2|


### Training Analysis
---
**📈 Loss Behaviour**

- **Training loss:**

    - 0.0996 (Epoch 1)

    - 0.0174 (Epoch 20)

- **Validation loss:**

    - Lowest at Epoch 11

    - Increased afterwards

**🎯 Key Observation**

- Best Validation IoU = 0.8764 at Epoch 11

- **After Epoch 11:**

    - Validation loss increased

    - Training loss kept decreasing

    - Clear sign of overfitting
Therefore, **Epoch 11 model selected for deployment.**

### Interfaces
---
### Subscribed Topics

| Topic Name       | Direction     | Message Type      | 
|-----------|-------------|-------------|
| `/camera/camera/color/image_raw` |Input| `sensor_msgs/Image` |

### Published Topics

| Topic Name       | Direction    | Message Type    | 
|-----------|-------------|-------------|
| `/road_seg/mask`| Output | `sensor_msgs/Image` |
| `/road_seg/overlay`| Output | `sensor_msgs/Image` |
