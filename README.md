# Hillfort Transformer — RGB+DTM semantic segmentation

This repository contains code and assets for automated detection and segmentation of small archaeological hillforts using fused aerial RGB orthoimagery and LiDAR-derived Digital Terrain Models (DTMs). The implementation adapts the CMX cross-modal fusion approach with transformer backbones (SegFormer / MiT) and includes data preparation, training, evaluation and baseline models.

## Abstract

Automated detection of small archaeological hillforts in large airborne datasets is challenging due to subtle topographic signatures and extreme class imbalance ($\sim$0.073\% positive pixels). We implemented a transformer-based semantic segmentation approach (RGBX with SegFormer backbone) that fuses aerial RGB imagery with LiDAR-derived digital terrain models via cross-modal feature integration (CMX). Our pipeline includes rasterisation of polygon labels, co-registration of RGB and DTM data, normalisation, and stratified tiling with oversampling to handle sparse positives. We evaluated on 118 test tiles covering 2.95B pixels. Baseline U-Net models achieved low training loss (0.01) but very poor hillfort detection performance (IoU=0.1\%, F1=0.2\%), demonstrating failure under extreme class imbalance. Transformer-based RGBX models showed substantial improvement: v2 and v3 achieved hillfort IoU of 14.7--15.7\% and F1 scores of $\sim$50\%. RGBX_v4 achieved significantly higher metrics (hillfort IoU=83.1\%, F1=90.7\%) but exhibited anomalous training behaviour (sudden 6x loss drop at epoch 34), making these results require validation. Results demonstrate that modern transformer architectures with cross-modal fusion can effectively handle extreme class imbalance in archaeological detection tasks, significantly outperforming traditional CNN baselines.

## Highlights

- **Multi-modal fusion (RGB + DTM):** dual-stream encoders with rectify-then-fuse cross-modal modules (CM-FRM + FFM).
- **Transformer backbones:** SegFormer (MiT) encoders (used: mit_b2 in experiments).
- **Baselines included:** U-Net variants (RGB-only and RGB+DTM) for comparison.
- **Class imbalance handling:** stratified tile buckets, sampling with replacement, per-tile/pixel weighting.
- **Preprocessing utilities:** rasterisation of polygons, tiling to 512×512, and dataset statistics generation.

## Repo structure (top-level)

- `RGBX/` — model code and configuration for RGB+DTM fusion[^1] (encoders, fusion modules, decoders, config).
- `baseline/`, `baseline2/` — baseline model implementations and training/evaluation scripts.
- `data/` — example and processed data artifacts (tile statistics, rasters, masks).
- `scripts/` — preprocessing, rasterisation and helper scripts used to prepare data and evaluate models.
- `notebooks/` — exploratory analysis and visualisation notebooks used during development.
- `runs/` — training run artifacts and results (predicted masks, logs, metrics).

## Data

- Input: paired 512×512 RGB tiles and single-channel DTM tiles (co-registered). Tiles produced from national geospatial rasters and vector polygon annotations.
- Labels: binary masks (0=background, 1=hillfort). The full test set comprises 118 raster tiles with an overall positive-pixel fraction ~0.08% (extreme class imbalance).

## Training & evaluation (summary from report)

- Main model: CMX-style RGBX with dual SegFormer-B2 encoders and an MLP decoder.
- Optimiser: AdamW ($\beta_1=0.9$, $\beta_2=0.999$, weight decay 0.01). The learning rate starts at $6 \times 10^{-5}$ and follows a polynomial warm-up/decay with a 10-epoch warm-up.
- Typical training config used in experiments: batch size 8, lr 6e-5, 40 epochs
- Loss: combination of Dice loss and cross-entropy (with dynamic class weights clipped to avoid instability).
- Evaluation metrics: pixel-level IoU, precision, recall, F1 (computed via sliding-window inference and optional ensembling).

Results reported in the project LaTeX report show substantial improvements of transformer RGBX models over U-Net baselines under extreme class imbalance; see `Report.pdf` for full experimental details and quantitative tables.

## Files to inspect

- `Report.pdf` — full project report with methodology, dataset details, experiments and results.

## Notes & caveats

- The dataset exhibits extreme class imbalance (positive pixels << 1%), so training uses stratified sampling with replacement and dynamic loss weighting. These strategies are implemented in the training pipeline and explained in `Report.pdf`.
- Some experimental runs exhibited anomalous behaviour (e.g., sudden loss drops) that require validation; treat exceptionally high reported metrics with caution until independently verified.

[^1]: Jiaming Zhang, Huayao Liu, Kailun Yang, Xinxin Hu, Ruiping Liu, and Rainer Stiefelhagen. CMX: Cross-Modal Fusion for RGB-X Semantic Segmentation With Transformers. IEEE Transactions on Intelligent Transportation Systems, 24(12):14679–14694, December 2023. Repository: https://github.com/huaaaliu/RGBX_Semantic_Segmentation.
