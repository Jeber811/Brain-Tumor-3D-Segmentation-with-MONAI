# Brain Tumor 3D Segmentation with MONAI

High-performance 3D Brain Tumor Segmentation pipeline using MONAI and PyTorch. This model processes multi-modal MRI data (FLAIR, T1, T1ce, T2) via a volumetric 3D U-Net to accurately segment Tumor Core, Whole Tumor, and Enhancing Tumor regions. 

Designed for clinical accuracy, the codebase features automated 3D transforms, rigorous Dice and HD95 evaluation metrics, and is heavily optimized for robust, persistent training within Kaggle notebook environments.

## Overview
The model is trained on the Medical Segmentation Decathlon (MSD) Task01 Brain Tumour dataset (`.nii` format). It processes 4-channel MRI scans to segment three overlapping tumor sub-regions based on the BraTS challenge classes:
1. **Tumor Core (TC)** (Labels 1 & 4)
2. **Whole Tumor (WT)** (Labels 1, 2, & 4)
3. **Enhancing Tumor (ET)** (Label 4)

## Model Architecture
* **Network:** Volumetric 3D U-Net (`monai.networks.nets.UNet`)
* **Structure:** `(16, 32, 64, 128, 256)` channels with strides of `(2, 2, 2, 2)` and 2 residual units.
* **Loss Function:** `DiceLoss` (utilizing squared prediction and sigmoid activation).
* **Optimization:** Adam optimizer ($lr = 1e-4$) combined with PyTorch AMP (`GradScaler`) for fast, mixed-precision training.

## Data Pipeline
The MONAI transform pipeline is designed to handle raw NIfTI files and includes:
* **Preprocessing:** RAS orientation alignment, `1.0x1.0x1.0` voxel spacing interpolation, and non-zero intensity normalization.
* **Augmentation:** Random $64 \times 64 \times 64$ spatial cropping, random flipping across all 3 spatial axes, and random intensity scaling/shifting to prevent overfitting.

## Training & Evaluation
* **Cross-Validation:** Robust 5-Fold Cross-Validation setup to ensure model generalizability.
* **Metrics:** Models are rigorously evaluated using the **Dice Metric** for volumetric area overlap and **Hausdorff Distance 95 (HD95)** for boundary precision.
* **Visualization:** Built-in Matplotlib utilities for plotting multi-channel FLAIR inputs alongside ground truth and predicted tumor masks for immediate qualitative analysis.

## Kaggle Environment Optimizations
This notebook includes specialized configurations to maximize efficiency and bypass common hardware limitations on Kaggle:
* **P100 GPU Compatibility:** Forces a downgrade to `torch==2.4.1` to restore full hardware compatibility for Kaggle's Tesla P100 accelerators.
* **Automated Checkpoint Recovery:** Automatically scans `/kaggle/input/` for previous `.pth` weights to resume training seamlessly, circumventing Kaggle's strict 12-hour session limits.
* **Clean Outputs:** Implements a custom context manager to suppress verbose C++ system-level noise (Google Logging) from TensorFlow/JAX during framework initialization.
