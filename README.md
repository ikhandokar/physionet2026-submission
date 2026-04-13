#  PhysioNet 2026 Challenge Submission  
**Future Cognitive Impairment Prediction from PSG Data**

---

##  Overview

This repository contains our submission for the **PhysioNet 2026 Challenge**, focusing on predicting future cognitive impairment using **polysomnography (PSG) data**.

The pipeline is designed to be:

- Modular and extensible  
- Robust to missing modalities  
- Reproducible with minimal setup  
- Efficient for inference on unseen data  

---

##  Method Summary

Our approach integrates multiple data sources:

- Physiological signals (EDF files)  
- Demographic features  
- Optional annotation features (CAISR)  
- Multi-modal fusion architecture  

The model is designed to handle real-world missing data scenarios using adaptive fusion strategies.

---

##  Repository Structure

```
physionet2026/
├── configs/
│   └── default.yaml
├── data/
│   ├── dataset.py
│   ├── edf_loader.py
│   ├── features.py
│   ├── annotations.py
│   └── channel_map.py
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
├── run_model.py
├── team_code.py
├── helper_code.py
└── requirements.txt
```

---

##  Expected Dataset Structure

>  Dataset is **NOT included** in this repository.

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
```

---

##  Environment Setup

**Tested on:** Python 3.10 (Windows/Linux), CUDA 12.1

### 1. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Install PyTorch

Install according to your system:

**GPU (CUDA 12.1):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CPU only:**
```bash
pip install torch torchvision torchaudio
```

For other configurations, see: https://pytorch.org/get-started/locally/

### 4. Verify Installation

```bash
python -c "import torch, numpy, pandas, sklearn, yaml; print('Environment OK')"
```

---

##  Configuration

Update dataset path in:

```
configs/default.yaml
```

Example:

```yaml
paths:
  data_root: "C:/path/to/PhysioNet/data"
```

---

##  Running Inference

Run the model on unseen data:

```bash
python run_model.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pt \
    --split supplementary_set
```

---


## ️ Training the Model

>  Training is optional for evaluation. Pretrained checkpoints can be used for inference.

### 1. Ensure Dataset is Prepared

```
PATH_TO_DATASET/
├── training_set/
│   ├── algorithmic_annotations/
│   ├── human_annotations/
│   ├── physiological_data/
│   └── demographics.csv
```

### 2. Start Training

```bash
python train_model.py --config configs/default.yaml
```

### 3. Checkpoints

```
checkpoints/
├── best_model.pt
├── latest_checkpoint.pt
```

### 4. Resume Training

```bash
python train_model.py \
    --config configs/default.yaml \
    --resume checkpoints/latest_checkpoint.pt
```

### 5. Training Logs

```
logs/
├── training_curves.png
├── metrics.json
```

### 6. Notes

- GPU is recommended for training  
- Hyperparameters are configurable in YAML  

---

##  Output

Generated outputs:

```
logs/
├── predictions/
│   └── *.csv
├── val_preview_predictions.json
└── training_curves.png
```

---

##  Important Notes

- Labels exist only in the training set  
- Supplementary set is used for inference  
- Missing modalities are handled dynamically  
- All behavior is controlled via YAML configuration  

---

##  Quick Start

```bash
git clone https://github.com/iftykhandokar/physionet2026-submission.git
cd physionet2026-submission

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python run_model.py --config configs/default.yaml --checkpoint checkpoints/best_model.pt
```

---

##  Contact

For questions, please open an issue in this repository.

---

##  License

This project is provided for research and competition purposes only.
