# 🎨AGD-GAN: Adaptive Gradient-Guided and Depth-supervised generative adversarial networks for ancient mural sketch extraction

[![Paper](https://img.shields.io/badge/Paper-Link-green)](https://www.sciencedirect.com/science/article/abs/pii/S0957417424015069)

## ⚙️Setup

We provide an environment.yml file listing the dependencies to create a conda environment. Our model uses PyTorch 1.7.1.
```bash
conda env create -f environment.yml
conda activate AGD-GAN
```

Use the following command to install CLIP (only needed for training).
```bash
conda activate AGD-GAN
pip install git+[https://github.com/openai/CLIP.git](https://github.com/openai/CLIP.git)
```
## 🧠 Training

To train a model named myexperiment from scratch, use the following command:
```bash
python train.py \
  --name Mural_model \
  --dataroot examples/train/photos \
  --depthroot examples/train/depthmaps \
  --root2 examples/train/drawings \
  --no_flip
```

Dataset Structure Requirements:

Replace the example data (examples/train/photos, examples/train/depthmaps, and examples/train/drawings) with the paths to your dataset of photographs, depth maps, and line drawings respectively.
Corresponding images and depth maps in the file paths specified by --dataroot and --depthroot must have the same file names.
You will also need to specify a path to an unaligned dataset of line drawings using --root2.
A small example of training data is provided in examples/train.
Training Tip: Because the model can start making grayscale photos after some training, it is highly recommended to save model checkpoints frequently by adding the flag --save_epoch_freq 1.
