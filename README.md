# WassersteinGAN_GP-PyTorch

### Update (Feb 21, 2020)

The mnist and fmnist models are now available. Their usage is identical to the other models: 
```python
from wgangp_pytorch import Generator
model = Generator.from_pretrained('g-mnist') 
```

### Overview
This repository contains an op-for-op PyTorch reimplementation of [Improved Training of Wasserstein GANs](http://xxx.itp.ac.cn/pdf/1704.00028).

The goal of this implementation is to be simple, highly extensible, and easy to integrate into your own projects. This implementation is a work in progress -- new features are currently being implemented.  

At the moment, you can easily:  
 * Load pretrained Generate models 
 * Use Generate models for extended dataset

_Upcoming features_: In the next few days, you will be able to:
 * Quickly finetune an Generate on your own dataset
 * Export Generate models for production

### Table of contents
1. [About Wasserstein GAN GP](#about-wasserstein-gan-gp)
2. [Model Description](#model-description)
3. [Installation](#installation)
4. [Usage](#usage)
    * [Load pretrained models](#loading-pretrained-models)
    * [Example: Extended dataset](#example-extended-dataset)
    * [Example: Visual](#example-visual)
5. [Contributing](#contributing) 

### About Wasserstein GAN GP

If you're new to Wasserstein GAN GP, here's an abstract straight from the paper:

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

Install from pypi:
```bash
pip install wgangp_pytorch
```

Install from source:
```bash
git clone https://github.com/Lornatang/WassersteinGAN_GP-PyTorch.git
cd WassersteinGAN_gp-PyTorch
pip install -e .
``` 

### Usage

#### Loading pretrained models

Load an Wasserstein GAN GP:
```python
from wgangp_pytorch import Generator
model = Generator.from_name("g-mnist")
```

Load a pretrained Wasserstein GAN GP:
```python
from wgangp_pytorch import Generator
model = Generator.from_pretrained("g-mnist")
```

#### Example: Extended dataset

As mentioned in the example, if you load the pre-trained weights of the MNIST dataset, it will create a new `imgs` directory and generate 64 random images in the `imgs` directory.

```python
import os
import torch
import torchvision.utils as vutils
from wgangp_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-mnist")
model.to(device)
# switch to evaluate mode
model.eval()

try:
    os.makedirs("./imgs")
except OSError:
    pass

with torch.no_grad():
    for i in range(64):
        noise = torch.randn(64, 100, device=device)
        fake = model(noise)
        vutils.save_image(fake.detach(), f"./imgs/fake_{i:04d}.png", normalize=True)
    print("The fake image has been generated!")
```

#### Example: Visual

```text
cd $REPO$/framework
sh start.sh
```

Then open the browser and type in the browser address [http://127.0.0.1:10003/](http://127.0.0.1:10003/).
Enjoy it.

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.   

I look forward to seeing what the community does with these models! 