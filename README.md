# StarCLR Inference

This repository provides **inference code** for [StarCLR](TODO:论文标题/链接), a Transformer-based contrastive pretraining framework for large-scale variable star light curves.

Note: This repo only contains the inference code. **Model weights are hosted externally** (see below).

---

## Download Model Weights
- [Google Drive Link](TODO:替换成实际链接)  
- [Alternative Release Assets](TODO:可选，放 GitHub Release 链接)

After downloading, place the checkpoint (e.g. `checkpoint/tess`) in a local path of your choice.

---

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
  --dataset ztf
```

This will output predicted labels and probabilities to output/.

## Repository Structure

```arduino
StarCLR-inference/
├─ src/
│  ├─ infer                                      
│  │  ├─ BertContrastiveLearningModel.py 
│  │  ├─ DataLoading.py 
│  │  ├─ Preprocess.py
│  │  └─ predict.py                          
├─ example/
│  ├─ example_tess.parquet 
│  ├─ example_ztf.parquet 
│  └─ example_gaia.parquet 
├─ requirements.txt
├─ README.md
├─ LICENSE
└─ .gitignore
```

## Citation

If you use this code, please cite:

```yaml
@article{TODO,
  title   = {StarCLR: Contrastive Pretraining for Variable Star Light Curves},
  author  = {TODO},
  journal = {TODO},
  year    = {2025},
}
```

## License

This project is released under the MIT License.
