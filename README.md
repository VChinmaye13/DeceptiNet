# DeceptiNet: Crafting and Catching Deepfakes with GAN Intelligence
DeceptiNet is an AI-driven deep learning project developed to accurately detect deepfake images by analyzing hidden artifacts and inconsistencies in facial features. The system aims to strengthen digital forensics and combat misinformation spread through AI-generated media.
A research-oriented starter scaffold for **deepfake detection** (and optional generation) using PyTorch.

## Quickstart

```bash
# 1) Create env (suggested)
conda create -n deceptinet python=3.10 -y
conda activate deceptinet

# 2) Install deps
pip install -r requirements.txt

# 3) Prepare data (ImageFolder layout)
# Put images into:
# data/processed/train/{real,fake}
# data/processed/val/{real,fake}
# data/processed/test/{real,fake}

# 4) Train detector
python -m src.detector.train_detector --config config.yaml

# 5) Evaluate
python -m src.detector.eval_detector --config config.yaml

# 6) Launch demo
python app/gradio_app.py --config config.yaml
```

## Folder Structure

```
DeceptiNet/
├── app/                  # Gradio demo
├── data/                 # raw + processed datasets
├── src/
│   ├── preprocessing/    # optional face extraction utilities
│   ├── detector/         # datasets, models, training, evaluation
│   └── generator/        # notes/placeholders for deepfake gen
├── checkpoints/          # saved models
├── logs/                 # training logs
├── notebooks/            # exploration
├── config.yaml           # central configuration
└── requirements.txt
```

## Datasets (suggested)
- FaceForensics++
- Celeb-DF
- DFDC

> Use `src/preprocessing/face_extract.py` if you want to crop faces from videos/images first.

## Ethics
This project is for **research and defense**: detecting synthetic media and studying robustness. Do not misuse.
