

```markdown
# PhysioNet 2026 Challenge Submission

A deep learning pipeline for future cognitive impairment prediction from polysomnography (PSG) data for the PhysioNet 2026 Challenge.

---

## Project Summary

This repository provides a modular and reproducible pipeline for processing PSG (EDF) data and performing risk prediction.

The model integrates:

- Physiological signals (EDF files)
- Demographic metadata
- Optional CAISR annotation features
- Multi-modal fusion with missing-modality robustness

---

## Repository Structure

```

physionet2026_v1/
├── configs/
│   └── default.yaml
├── data/
│   ├── annotations.py
│   ├── channel_map.py
│   ├── dataset.py
│   ├── edf_loader.py
│   └── features.py
├── models/
│   ├── encoders.py
│   ├── fusion.py
│   └── model.py
├── utils/
│   ├── io.py
│   ├── logger.py
│   ├── metrics.py
│   └── seed.py
├── checkpoints/
├── logs/
├── requirements.txt
├── train_model.py
├── run_model.py
├── team_code.py
└── helper_code.py

```

---

## Expected Dataset Structure

Dataset is NOT included.

```

PATH_TO_DATASET/
├── training_set/
│   ├── algorithmic_annotations/
│   ├── human_annotations/
│   ├── physiological_data/
│   └── demographics.csv
└── supplementary_set/
├── physiological_data/
└── demographics.csv

```

Example file:
```

sub-S0001111197789_ses-2.edf

````

---

## Environment Setup (Windows)

Create virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
````

Install dependencies:

```bat
pip install -r requirements.txt
```

Install PyTorch (CUDA 12.1):

```bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:

```bat
python -c "import numpy, torch, pandas, sklearn, yaml; print('Environment OK')"
```

---

## Dataset Path Setup

Open:

```
configs/default.yaml
```

Update:

```yaml
paths:
  data_root: "PATH_TO_DATASET"
```

Example:

```yaml
paths:
  data_root: "C:/Users/YourName/Desktop/Physionet-2026/archive"
```

---

## Run the Model

Run pipeline:

```bat
python train_model.py --config configs/default.yaml
```

Run inference:

```bat
python run_model.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt --split supplementary_set
```

---

## Outputs

Generated outputs:

```
checkpoints/best_model.pt
checkpoints/latest_checkpoint.pt
logs/training_curves.png
logs/val_preview_predictions.json
logs/predictions/*.csv
```

---

## Notes

* Labels exist ONLY in training set
* Supplementary set → labels will be NULL (expected)
* Model supports dynamic missing modalities
* Configurable via YAML file

---

## Quick Start

```
git clone https://github.com/iftakharAK/physionet2026-submission.git
cd physionet2026-submission

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python train_model.py --config configs/default.yaml
```

---

## Future Improvements

* Domain generalization
* Better signal augmentation
* Improved cross-site robustness
* Calibration tuning


