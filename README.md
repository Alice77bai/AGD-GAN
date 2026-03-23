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
## 🧪 Testing

Run the pre-trained model on images in --dataroot. Replace examples/test with the folder path containing your input images.
```bash
python test.py --name anime_style --dataroot examples/test
```

### 📁 Dataset Requirements

Before you kick off the training, make sure your dataset is organized according to these guidelines:

* **Swap out the paths:** First, you'll need to replace the default example paths (`examples/train/photos`, `examples/train/depthmaps`, and `examples/train/drawings`) with the actual local directories where your photographs, depth maps, and line drawings are stored.
* **⚠️ Filenames must match exactly:** For the corresponding image pairs in the folders specified by `--dataroot` (photos) and `--depthroot` (depth maps), they **must have the exact same filenames**. Don't mix them up!
* **Specify unaligned data:** Additionally, you'll need to use the `--root2` parameter to point the model to an unaligned dataset of line drawings.

> 💡 **Training Tip:** As training progresses, the model might occasionally get a bit "lazy" and start outputting grayscale photos instead of pure line drawings. To avoid losing your hard work, we highly recommend adding the `--save_epoch_freq 1` flag to your command. This forces the model to save checkpoints more frequently, making it much easier for you to pick out the absolute best version later!

### 🗺️ Depth Maps

When it comes to training, teaching the model about geometry means we need depth maps to pair with your dataset of photographs. 
To generate these pseudo-ground truth depth maps, we lean on an awesome pretrained model from [Boosting Monocular Depth Estimation Models to High-Resolution via Content-Adaptive Multi-Resolution Merging.](https://yaksoy.github.io/highresdepth/).

###📜 Citation
If you find this work useful please use the following citation:
```bash
@article{yu2024agd,
  title={AGD-GAN: Adaptive Gradient-Guided and Depth-supervised generative adversarial networks for ancient mural sketch extraction},
  author={Yu, Zhe and Peng, Shenglin and Qu, Shuyi and Zhang, Qunxi and Wang, Jun and Peng, Jinye},
  journal={Expert Systems with Applications},
  volume={255},
  pages={124639},
  year={2024},
  publisher={Elsevier}
}
```

## 🤝 Acknowledgements
Model code adapted from [pytorch-CycleGAN-and-pix2pix](pytorch-CycleGAN-and-pix2pix) and [Informative Drawings: Learning to generate line drawings that convey geometry and semantics](https://github.com/carolineec/informative-drawings)

