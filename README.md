# Thermal-Tracking-DETR: Training

Fine-tunes a DETR object detector (`facebook/detr-resnet-50` from Hugging Face) on the
**Teledyne FLIR ADAS v2** thermal dataset so it can find vehicles and pedestrians in
infrared images.

This repository holds the training code. The trained model is served by a FastAPI backend
and a Streamlit dashboard in **[Thermal_web_app](https://github.com/Hamza237516/Thermal_web_app)**.

## How it works

- **Model:** DETR with a ResNet-50 backbone, pretrained on COCO (ordinary color images) and
  fine-tuned on thermal frames.
- **Data:** the FLIR ADAS v2 thermal training split, read from its COCO-format `coco.json`.
  Thermal frames are single-channel; they are loaded as 3-channel images so they match the
  input the pretrained backbone expects.
- **Labels:** FLIR uses COCO-style category IDs, so the model keeps DETR's 91-label COCO
  classification head instead of a smaller one.
- **Loss:** DETR's built-in Hungarian-matching loss (classification, box L1 and GIoU terms).

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads FLIR images and annotations in the format DETR's image processor expects |
| `train.py` | Training loop; saves a checkpoint after every epoch |
| `test_dataset.py` | Checks that the dataset loads and prints the first sample |
| `requirements.txt` | Python dependencies |

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Hamza237516/Thermal-Tracking-DETR.git
   cd Thermal-Tracking-DETR
   ```
2. Install the dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the [FLIR ADAS thermal dataset](https://oem.flir.com/solutions/automotive/adas-dataset-form/)
   (v2) and unzip it so the training split sits at
   `extracted_flir/FLIR_ADAS_v2/images_thermal_train/`, the folder that contains `coco.json`.

## Training

Check the setup first. These load the data and run 20 training steps without saving anything:
```bash
python test_dataset.py
python train.py --smoke-test
```

Full fine-tuning (a GPU is strongly recommended):
```bash
python train.py
```

| Setting | Default | Flag |
|---|---|---|
| Epochs | 15 | `--epochs` |
| Batch size | 2 | `--batch-size` |
| Optimizer | AdamW | |
| Learning rate | 1e-4 | `--lr` |
| Data folder | `extracted_flir/FLIR_ADAS_v2` | `--data-dir` |
| Checkpoint folder | `checkpoints/` | `--output-dir` |
| Device | CUDA if available, otherwise CPU | `--device` |

Each epoch saves `checkpoints/thermal_detr_epoch_<N>.pth`.

## Using the trained model

Copy `checkpoints/thermal_detr_epoch_15.pth` into the Thermal_web_app folder. Its FastAPI
backend loads that file and serves detections to the dashboard. There, the fine-tuned model
takes about 0.45 s per image on a MacBook CPU and localizes 35+ objects in a single frame.

---
Developed by [Hamza Mehmood](https://github.com/Hamza237516), B.Tech student at NIT Srinagar
