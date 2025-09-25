# StarCLR Inference

This repository provides **inference code** for [StarCLR](TODO:论文标题/链接), a Transformer-based contrastive pretraining framework for large-scale variable star light curves.

Note: This repo only contains the inference code. **Model weights are hosted externally** (see below).

## Download Model Weights
- [Google Drive Link](https://drive.google.com/drive/folders/1eKim9iKv4NIjoKlUwS2ktLduKH4Pl3vX?usp=drive_link)  

After downloading, place the model weights into the `checkpoints/` directory.

## Download Example Data
- [Google Drive Link](https://drive.google.com/drive/folders/1Bx2NnwzYgb7ZBSNHKXSLEgW6i7TG3jC5?usp=drive_link)  

After downloading, place the example parquet files into the `example/` directory:

## Installation
Clone this repo and install dependencies:

```bash
git clone https://github.com/TODO/StarCLR-inference.git
cd StarCLR-inference
pip install -r requirements.txt
```

Note on PyTorch installation:

In requirements.txt, the PyTorch installation line is commented out:
  
```text
# torch==2.7.0+cu126
```

- If you want to use the CPU version, simply uncomment this line.

- If you want to use the GPU version with CUDA 12.6, install it separately with:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

## Quick Start

Run inference on the example data:

```bash
python src/infer/predict.py \
  --dataset 
```

This will output predicted labels and probabilities to output/.

## Repository Structure

```arduino
StarCLR-inference/
├─ src/
│  └─ infer/
│     ├─ BertContrastiveLearningModel.py   # StarCLR model definition (adapted from Transformers)
│     ├─ DataLoading.py                    # Dataset wrapper and collate function for batching
│     ├─ Preprocess.py                     # Preprocessing
│     └─ predict.py                        # Inference entry script
│
├─ example/                                # Example data directory
│  ├─ example_gaia.parquet                 # Small demo dataset (Gaia sample)
│  ├─ example_ztf.parquet                  # Small demo dataset (ZTF sample)
│  └─ example_tess.parquet                 # Small demo dataset (TESS sample)
│
├─ checkpoints/                            # Model checkpoints directory
│  ├─ gaia/
│  │  ├─ model.safetensors                 # Gaia model weights
│  │  └─ config.json                       # Gaia model config
│  ├─ tess/
│  │  ├─ model.safetensors                 # TESS model weights
│  │  └─ config.json                       # TESS model config
│  └─ ztf/
│     ├─ model.safetensors                 # ZTF model weights
│     └─ config.json                       # ZTF model config
│
├─ outputs/                                # Inference results directory
│
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore
```

<!-- ## Citation

If you use this code, please cite:

```yaml
@article{TODO,
  title   = {StarCLR: Contrastive Learning Representation for Astronomical Light Curves},
  author  = {TODO},
  journal = {TODO},
  year    = {2025},
}
``` -->

## License

This project is released under the MIT License.
