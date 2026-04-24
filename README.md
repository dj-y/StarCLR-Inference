# StarCLR Inference

This repository provides inference code for StarCLR, a Transformer-based contrastive pretraining framework for large-scale variable star light curves.

Note: this repo contains inference code only. Model weights and example data are hosted externally.

## Download Assets

- Google Drive (weights + example data):
  https://drive.google.com/drive/folders/1hNlb4Ulsd9nIfXJsLEAy6hem3XFv2i7V

After downloading:

- put model checkpoints into `checkpoints/`
- put example parquet files into `example/`

## Installation

```bash
git clone https://github.com/dj-y/StarCLR-inference.git
cd StarCLR-inference
pip install -r requirement.txt
```

PyTorch note:

- `requirement.txt` keeps the torch line commented:
  ```text
  # torch==2.7.0+cu126
  ```
- install the CUDA 12.6 build manually if needed:
  ```bash
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  ```

## Quick Start

1. Stage-1: extract hidden representations

```bash
python src/infer/HiddenStates.py --dataset tess --batch_size 16
```

2. Stage-2: run prediction head

```bash
python src/infer/Predict.py --dataset tess --batch_size 16
```

Optional device override for both scripts:

```bash
--device cpu
# or
--device cuda:0
```

Outputs:

- Stage-1: `output/hidden_states_<dataset>.parquet`
- Stage-2: `output/predictions_label_<dataset>.csv`

## Repository Structure

```text
StarCLR-inference/
├── src/
│   └── infer/
│       ├── BertContrastiveLearningModel.py
│       ├── DataLoading.py
│       ├── HiddenStates.py
│       ├── Predict.py
│       └── Preprocess.py
├── checkpoints/
├── example/
├── output/
├── requirement.txt
├── README.md
└── LICENSE
```

## License

This project is primarily released under the MIT License.
Some source files include code derived from Hugging Face Transformers under Apache License 2.0.
See `LICENSE` for full details.
