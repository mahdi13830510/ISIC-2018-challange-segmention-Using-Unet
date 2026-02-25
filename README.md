# UNet Skin Lesion Segmentation -- ISIC 2018

A TensorFlow/Keras implementation of UNet for binary skin lesion segmentation, trained and evaluated on the ISIC 2018 Challenge Task 1 dataset.

## Results

| Metric | Score |
|---|---|
| Dice | 0.9211 |
| IoU | 0.8570 |
| Accuracy | 0.8967 |
| Precision | 0.9106 |
| Recall | 0.9363 |
| F1 | 0.9233 |

## Architecture

Standard UNet encoder-decoder with skip connections. Filter progression: 32 - 64 - 128 - 256 - 512 (bottleneck). Loss is a combination of binary cross-entropy and Dice loss to handle the foreground/background class imbalance typical in dermoscopy images.

## Notebook contents

1. **EDA** -- image dimension distributions, lesion coverage statistics, class balance, and sample visualisations.
2. **Model and training** -- UNet definition, tf.data pipeline with augmentation, Adam optimizer with ReduceLROnPlateau and early stopping over 50 epochs.
3. **Evaluation** -- per-metric bar charts, per-sample Dice distributions, prediction overlays, error maps, and confidence histograms.

## Setup

The notebook is designed to run on Kaggle with GPU enabled. Add the [ISIC 2018 Task 1 dataset](https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation) to your notebook, confirm the image and mask paths in the `CFG` dictionary, and run all cells.

## Requirements

- Python 3.10+
- TensorFlow 2.19+
- NumPy, OpenCV, Matplotlib
