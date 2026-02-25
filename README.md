<div align="center">
  <h1 style="font-size: 36px;">Road Mapping Using Sementic Segmentation</h1>
</div>

## Table Of Contents
- [Overview](#overview)
- [Research Question](#research-question)
- [Hyperparameters](#hyperparameters)
- [Training Strategy](#training-strategy)
- [Training Analysis](#training-analysis)
- [Model Selection](#model-selection)
- [Interfaces](#interfaces)
- [Deployment Features](#deployment-features)
- [Conclusion](#conclusion)

---

## Overview

Road perception is essential for intelligent vehicles. However, HD-map based and multi-sensor approaches are expensive and difficult to maintain in small-scale environments.

This project explores a camera-only road mapping pipeline using semantic segmentation to extract lane geometry from RGB images captured by an Intel RealSense camera.

Instead of relying on BEV-based complex systems like LaneSegNet, we use a pixel-wise segmentation model (DeepLabV3+) to directly detect drivable lane areas and extract geometry from segmentation masks.

The final system is deployed in ROS 2 using an ONNX-optimized model for real-time inference in a Model City environment.

---

## Research Question

**“How well can semantic segmentation be applied to RGB images from Intel RealSense camera to extract lane geometry in a model city environment?”**

---

## Hyperparameters

| Parameter        | Value                                  | Reason (Why this was chosen) |
|------------------|----------------------------------------|-------------------------------|
| **Model**        | DeepLabV3+                             | Captures multi-scale context and works well for thin lane structures |
| **Encoder**      | ResNet-50 (ImageNet pretrained)        | Pretrained backbone improves feature extraction and speeds up convergence |
| **Framework**    | PyTorch + segmentation_models_pytorch  | Stable research framework with ready-made segmentation implementations |
| **Input Size**   | 512 × 512                              | Preserves lane detail while fitting GPU memory efficiently |
| **Classes**      | 1 (Lane)                               | Binary segmentation simplifies training and improves stability |
| **Batch Size**   | 4                                      | Balanced for GPU memory at 512 resolution |
| **Epochs**       | 20                                     | Sufficient for convergence without excessive training time |
| **Optimizer**    | Adam                                   | Adaptive optimizer providing stable training |
| **Learning Rate**| 1e-4                                   | Safe standard learning rate for Adam with pretrained backbone |
| **Loss**         | CrossEntropyLoss                       | Suitable for pixel-wise classification tasks |
| **Train Loader** | shuffle=True, workers=2                | Shuffle improves generalization; workers speed up data loading |
| **Val Loader**   | shuffle=False, workers=2               | Ensures consistent validation evaluation |

---

## Training Strategy

The model was trained **three times**, each time using an improved dataset to handle real-world challenges observed in the Model City.

---

### 🔹 Training 1 – Baseline Model

**Problem Observed:**
- Lower accuracy in shadow regions
- Weak performance in low-light areas
- Confusion between lane and background under reflections

**Reason:**
The initial dataset mainly contained normal lighting conditions.  
The model struggled when contrast was low or shadows overlapped lane markings.

**Result:**
- Good segmentation in normal lighting
- Poor generalization under shadows and dim conditions

---

### 🔹 Training 2 – Lighting Diversity Improvement

**Dataset Expanded With:**
- Bright sunlight
- Afternoon without artificial lights
- Model City artificial lighting
- Night-like indoor conditions

**Why These Scenarios Were Chosen:**
Lane detection depends heavily on illumination and contrast.  
Adding lighting diversity improves robustness and reduces brightness sensitivity.

**Result:**
- Improved performance across lighting variations
- More stable segmentation in sunlight and night conditions

---

### 🔹 Training 3 – Model City Edge Case Refinement

**Problem Observed:**
- Lane and non-drivable areas had similar color
- Same white divider lines used across different zones
- Model confused drivable and non-drivable regions

**Dataset Enhancement:**
- Added targeted samples from problematic areas
- Included difficult frames with similar textures
- Improved annotation precision

**Why This Was Important:**
When color and texture are similar, the model must rely more on spatial context and geometry patterns.

**Result:**
- Reduced glitch behavior
- Improved centerline extraction
- More stable segmentation in same-color regions

---
## Training Analysis

This section explains how the segmentation model was evaluated during training and why a specific checkpoint was selected for deployment.

---

### 1️⃣ Intersection over Union (IoU)

IoU (Intersection over Union) is the primary metric used to evaluate segmentation accuracy.

It measures how well the predicted lane mask overlaps with the ground-truth annotation.

Conceptually:

IoU = Overlapping Area / Total Combined Area

More specifically:

- **Intersection** → Pixels correctly predicted as lane  
- **Union** → All pixels that are either predicted as lane or actually lane  

This means:

- IoU = 0 → No overlap at all  
- IoU = 1 → Perfect segmentation  

Since this is a binary segmentation task (lane vs background), IoU directly reflects how accurately the drivable region is extracted.

---

### 2️⃣ How IoU is Computed During Validation

During each validation phase:

1. The model generates a predicted binary mask.
2. The predicted mask is compared pixel-by-pixel with the ground truth mask.
3. Pixels where both are lane → counted as intersection.
4. Pixels belonging to either prediction or ground truth → counted as union.
5. IoU is computed per image.
6. The average IoU across the validation set is calculated.

This average IoU becomes the key performance indicator for model selection.

---

### 3️⃣ Best Observed Performance

The highest validation IoU achieved:

**0.8764 at Epoch 11**

This means that approximately **87.64% of the predicted lane area overlapped correctly with the ground truth**.

At this point:
- The model generalized well.
- The predicted lane masks were visually aligned with annotations.
- Boundaries were stable and consistent.

---

### 4️⃣ Loss Behaviour Analysis

#### Training Loss

- 0.0996 (Epoch 1)
- 0.0174 (Epoch 20)
- Decreased continuously

This indicates that the model kept learning patterns from the training dataset and fitting the data more closely.

---

#### Validation Loss

- Lowest at Epoch 11
- Increased gradually after Epoch 11
- Reached 0.0551 at Epoch 20

Validation loss reflects performance on unseen data.  

While training loss kept decreasing, validation loss started increasing after Epoch 11.

This divergence indicates that the model began to memorize the training data instead of improving generalization.

---

### 5️⃣ Overfitting Observation

After Epoch 11:

- Training loss ↓ (improving on training data)
- Validation loss ↑ (worsening on unseen data)
- Validation IoU stopped improving

This pattern clearly indicates **overfitting**.

The model was learning fine details specific to the training set but not improving performance on new data.

Because:

- Epoch 11 had the highest validation IoU (0.8764)
- Validation loss was at its minimum
- Generalization was strongest at this point

The checkpoint from **Epoch 11** was selected for deployment.

Further training beyond Epoch 11 did not improve segmentation quality and instead reduced generalization capability.

---

## Model Selection

Due to overfitting beyond Epoch 11, the model from **Epoch 11** was selected for deployment.

**Deployment Model:**
- Best validation IoU
- Lowest validation loss
- Best generalization performance

Saved as: deeplabv3plus_road_best.pth
deeplabv3plus_road_best.pth
Converted to ONNX: 
deeplabv3plus_road.onnx


Used for real-time ROS 2 inference.

---
# Core Computation: Masking & Depth-Based Projection

This section explains the internal computational logic behind:

- Lane mask generation  
- Depth filtering using the mask  
- Lane width estimation  
- Pixel-to-meter projection  

Only the core operations used in the implementation are described.

---

## 1️⃣ Lane Mask Generation

After ONNX inference, the segmentation model outputs a tensor of shape:

[Batch, Classes, Height, Width]

For each pixel:

- The network produces raw scores (logits) for two classes:
  - Background
  - Lane
- The class with the highest score is selected using `argmax`.
- Pixels classified as lane are assigned value **1**.
- Background pixels are assigned value **0**.

This produces a **binary segmentation mask**.

The mask is:

- Resized back to original resolution (if required)
- Converted to a green overlay for visualization
- Blended with the original RGB image

This binary mask becomes the foundation for all geometric and depth computations.

---

## 2️⃣ Depth Filtering Using the Lane Mask

The depth image is received as raw sensor data (typically 16-bit values in millimeters).

### Processing Steps:

1. Convert depth from millimeters to meters  
   (ensures real-world metric units)

2. Resize the segmentation mask to match depth resolution.

3. Apply logical filtering:
   - Only pixels where mask == lane are kept.
   - All background depth pixels are ignored.

This ensures:

- Distance measurements are calculated only within the drivable lane.
- Background objects (walls, buildings, vehicles) do not affect measurements.
- Noise is significantly reduced.

To improve robustness:
- Median depth is used instead of mean.
- This reduces influence of outliers and sensor spikes.

---

## 3️⃣ Lane Width Estimation (Pixel Space)

Lane width is first computed in image space.

At a selected horizontal row:

- Find the leftmost pixel classified as lane.
- Find the rightmost pixel classified as lane.
- Compute:

width_px = right_x − left_x

This gives the lane width in pixels.

This step is purely image-based and does not yet represent real-world size.

---

## 4️⃣ Pixel-to-Meter Projection

To convert pixel width into real-world meters, the pinhole camera projection principle is used.

The relationship between image coordinates and real-world width depends on:

- Pixel width
- Depth (distance from camera)
- Camera focal length (fx)

Projection concept:

Real Width ≈ (Pixel Width × Depth) / Focal Length

Where:

- Pixel Width → width_px  
- Depth → median depth value at that row  
- Focal Length → fx (camera intrinsic parameter)

This converts 2D image measurement into a 3D metric estimate.

---

## 5️⃣ Bottom 10% Region Strategy

Instead of calculating width anywhere in the image:

- Only the bottom 10% of the image is considered.
- This region represents the nearest visible part of the lane.

Why this improves stability:

- Near-field depth is more reliable.
- Perspective distortion is smaller.
- Width estimation becomes physically more accurate.
- Reduces influence of far noisy pixels.

This design choice significantly improves real-world lane width estimation stability.

---

## Core Implementation Summary

- Segmentation produces a binary lane mask using argmax.
- Mask filters the depth image to isolate drivable region.
- Lane width is first computed in pixels.
- Pixel width is projected into meters using camera intrinsics.
- Bottom-region selection improves measurement robustness.

This combination of deep learning segmentation and geometric projection forms the measurement backbone of the road mapping system.
## Interfaces

### Subscribed Topics

| Topic Name | Direction | Message Type | Information Contained |
|------------|----------|--------------|------------------------|
| `/camera/camera/color/image_raw` | Input | `sensor_msgs/Image` | Raw RGB image frames from Intel RealSense camera used for semantic segmentation inference |
| `/camera/camera/depth/image_rect_raw` | Input | `sensor_msgs/Image` | Raw depth image (16UC1) used for distance estimation and lane width calculation |

---

### Published Topics

| Topic Name | Direction | Message Type | Information Contained |
|------------|----------|--------------|------------------------|
| `/mask` | Output | `sensor_msgs/Image` | Binary lane segmentation mask (green lane region on black background) |
| `/overlay` | Output | `sensor_msgs/Image` | Original RGB image overlaid with semi-transparent green lane mask |
| `/depth_masked` | Output | `sensor_msgs/Image` | Depth colormap visualized only on segmented lane area for distance and width estimation |

---

## Deployment Features

- ONNX inference using CPUExecutionProvider
- Real-time lane segmentation
- RGB overlay visualization
- Lane width estimation in meters
- Near and far lane visibility estimation using depth
- Bottom 10% region lane width measurement for practical estimation

---

## Conclusion

- Camera-based semantic segmentation successfully extracts drivable lane regions in Model City.
- Achieved IoU of 0.8764 at best epoch.
- Lane geometry was directly derived from segmentation masks.
- System deployed in ROS 2 for real-time operation.
- Demonstrates that camera-only perception can effectively perform road mapping in structured small-scale environments.
