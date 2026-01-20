# Hillfort Transformer: RGB-X Semantic Segmentation for Hillfort Detection

A deep learning project for automated detection and segmentation of hillforts from aerial imagery using multi-modal RGB and Digital Terrain Model (DTM) data. This repository implements state-of-the-art semantic segmentation architectures with cross-modal fusion capabilities.

## Project Overview

This project combines **RGB orthoimagery** and **Digital Terrain Models (DTM)** to perform binary semantic segmentation for hillfort detection. The system uses transformer-based encoders (SegFormer, Swin Transformer) with specialized decoders optimized for detecting archaeological features in topographic data.

### Key Features

- **Multi-modal Input**: Fuses RGB imagery with DTM auxiliary data for improved segmentation
- **Transformer Architectures**: Supports SegFormer (mit_b0-b5) and Swin Transformer backbones
- **Multiple Decoders**: Includes MLPDecoder, UPernet, and DeepLabV3+ decoders
- **Stratified Training**: Balances positive/negative samples with stratified bucket-based sampling
- **Distributed Training**: Multi-GPU support via PyTorch DistributedDataParallel
- **Automated Data Preprocessing**: Scripts for rasterizing vector geometries, tiling datasets, and computing statistics
- **Class Imbalance Handling**: Pixel-level weighting and tile-level oversampling strategies


## Configuration (`config.py`)

The [RGBX/config.py](RGBX/config.py) file centralizes all training hyperparameters:

### Dataset Configuration
- **Dataset Name**: `HillfortDataSet`
- **Image Format**: 512×512 PNG tiles (RGB)
- **Ground Truth**: Single-channel PNG masks (0=background, 1=hillfort)
- **Auxiliary Input (DTM)**: Single-channel GeoTIFF files

### Model Architecture
- **Backbone**: SegFormer-B2 (mit_b2) - transformer-based encoder
- **Decoder**: MLPDecoder
- **Input Channels**: 4 (RGB + DTM stacked)
- **Output**: 2 classes (background, hillfort)
- **Pretrained Weights**: SegFormer mit_b2.pth

### Training Hyperparameters
- **Batch Size**: 8
- **Learning Rate**: 6e-5 (with cosine annealing)
- **Optimizer**: AdamW with weight decay 0.01
- **Epochs**: 20
- **Warm-up Epochs**: 10
- **Loss Function**: DiceCrossEntropyLoss (dice_weight=1.0, ce_weight=1.0)

### Class Imbalance Handling
- **Stratified Bucket Sampling**: Divides tiles into buckets by positive pixel fraction
  - `neg`: 0% positives
  - `small`: 0% < f ≤ 0.1%
  - `mid`: 0.1% < f ≤ 1%
  - `full`: f > 1%
- **Bucket Proportions**: [0.5, 0.25, 0.15, 0.10] (configurable)
- **Positive Oversampling**: 3× duplication of positive bucket entries
- **Pixel-Level Weighting**: Computed from tile_stats.csv for precise class balancing
- **Max Class Weight Clipping**: 50.0 (prevents extreme weights)

### Data Augmentation
- **Random Scaling**: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
- **Evaluation Scaling**: 1.0 (no test-time augmentation)
- **Evaluation Stride**: 2/3 (sliding window with 33% overlap)
- **Normalization Mean**: [0.485, 0.456, 0.406] (ImageNet standard)
- **Normalization Std**: [0.229, 0.224, 0.225]

## Dataset Format

### File Format Details
- **RGB**: PNG files, 8-bit RGB (3 channels), 512×512
- **Labels**: PNG files, 8-bit grayscale (1 channel), 0 or 1, 512×512
- **DTM**: GeoTIFF files, 32-bit float (single channel), 512×512
- **train.txt / test.txt**: Text files with tile basenames (no extension)

## Model Checkpoints

### Available Backbones
- **SegFormer**: mit_b0 (32M params), mit_b1, mit_b2, mit_b4, mit_b5 (82M params)
- **Swin Transformer**: swin_s, swin_b

### Decoder Options
- **MLPDecoder**: Lightweight, efficient MLP-based fusion
- **UPernet**: Pyramid pooling module for multi-scale features
- **DeepLabV3+**: Atrous spatial pyramid pooling