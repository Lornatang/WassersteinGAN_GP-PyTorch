# Wasserstein_GP-PyTorch

### Overview

This repository contains an op-for-op PyTorch reimplementation
of [Improved Training of Wasserstein GANs](http://xxx.itp.ac.cn/abs/1704.00028).

### Table of contents

1. [About Improved Training of Wasserstein GANs](#about-improved-training-of-wasserstein-gans)
2. [Model Description](#model-description)
3. [Installation](#installation)
    * [Clone and install requirements](#clone-and-install-requirements)
    * [Download pretrained weights](#download-pretrained-weights-eg-mnist)
4. [Test](#test)
5. [Train](#train-eg-mnist)
6. [Contributing](#contributing)
7. [Credit](#credit)

### About Improved Training of Wasserstein GANs

If you're new to WGAN-GP, here's an abstract straight from the paper:

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The
recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate
only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in
WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to
clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs
better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no
hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality
generations on CIFAR-10 and LSUN bedrooms.

### Model Description

We have two networks, G (Generator) and D (Discriminator).The Generator is a network for generating images. It receives
a random noise z and generates images from this noise, which is called G(z).Discriminator is a discriminant network that
discriminates whether an image is real. The input is x, x is a picture, and the output is D of x is the probability that
x is a real picture, and if it's 1, it's 100% real, and if it's 0, it's not real.

### Installation

#### Clone and install requirements

```bash
$ git clone https://github.com/Lornatang/WassersteinGAN_GP-PyTorch.git
$ cd WassersteinGAN_GP-PyTorch/
$ pip3 install -r requirements.txt
```

#### Download pretrained weights (e.g. mnist)

```bash
$ cd weights/
$ python3 download_weights.py
```

### Test

Using pre training model to generate pictures.

```text
usage: test.py [-h] [-a ARCH] [-n NUM_IMAGES] [--outf PATH] [--device DEVICE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  -a ARCH, --arch ARCH  model architecture: mnist | fashion_mnist |cifar10 |
                        (default: mnist)
  -n NUM_IMAGES, --num-images NUM_IMAGES
                        How many samples are generated at one time. (default:
                        64).
  --outf PATH           The location of the image in the evaluation process.
                        (default: ``test``).
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default:
                        ``cpu``).

# Example (e.g. MNIST)
$ python3 test.py -a mnist
```

<span align="center"><img src="assets/mnist.gif" alt="">
</span>

### Train (e.g. MNIST)

```text
usage: train.py [-h] --dataset DATASET [--dataroot DATAROOT] [-j N]
                [--manualSeed MANUALSEED] [--device DEVICE] [-p N] [-a ARCH]
                [--pretrained] [--netD PATH] [--netG PATH] [--start-epoch N]
                [--iters N] [-b N] [--image-size IMAGE_SIZE]
                [--channels CHANNELS] [--lr LR] [--n_critic N_CRITIC]
                [--clip_value CLIP_VALUE]

Research and application of GAN based super resolution technology for
pathological microscopic images.

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist | fashion-mnist | cifar10 |.
  --dataroot DATAROOT   Path to dataset. (default: ``data``).
  -j N, --workers N     Number of data loading workers. (default:4)
  --manualSeed MANUALSEED
                        Seed for initializing training. (default:1111)
  --device DEVICE       device id i.e. `0` or `0,1` or `cpu`. (default: ````).
  -p N, --save-freq N   Save frequency. (default: 50).
  -a ARCH, --arch ARCH  model architecture: mnist | fashion_mnist |cifar10 |
                        (default: mnist)
  --pretrained          Use pre-trained model.
  --netD PATH           Path to latest discriminator checkpoint. (default:
                        ````).
  --netG PATH           Path to latest generator checkpoint. (default: ````).
  --start-epoch N       manual epoch number (useful on restarts)
  --iters N             The number of iterations is needed in the training of
                        PSNR model. (default: 5e5)
  -b N, --batch-size N  mini-batch size (default: 64), this is the total batch
                        size of all GPUs on the current node when using Data
                        Parallel or Distributed Data Parallel.
  --image-size IMAGE_SIZE
                        The height / width of the input image to network.
                        (default: 28).
  --channels CHANNELS   The number of channels of the image. (default: 1).
  --lr LR               Learning rate. (default:0.0002)
  --n_critic N_CRITIC   Number of training steps for discriminator per iter.
                        (Default: 5).
  --clip_value CLIP_VALUE
                        Lower and upper clip value for disc. weights.
                        (Default: 0.01).

# Example (e.g. MNIST)
$ python3 train.py -a mnist --dataset mnist --image-size 28 --channels 1 --pretrained
```

If you want to load weights that you've trained before, run the following command.

```bash
$ python3 train.py -a mnist \
                   --dataset mnist \
                   --image-size 28 \
                   --channels 1 \
                   --start-epoch 18 \
                   --netG weights/netG_epoch_18.pth \
                   --netD weights/netD_epoch_18.pth
```

### Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### Improved Training of Wasserstein GANs

*Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville*

**Abstract**

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The
recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate
only low-quality samples or fail to converge. We find that these problems are often due to the use of weight clipping in
WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to
clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs
better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no
hyperparameter tuning, including 101-layer ResNets and language models over discrete data. We also achieve high quality
generations on CIFAR-10 and LSUN bedrooms.

[[Paper]](http://xxx.itp.ac.cn/abs/1712.01026)

```
@article{DBLP:journals/corr/GulrajaniAADC17,
  author    = {Ishaan Gulrajani and
               Faruk Ahmed and
               Mart{\'{\i}}n Arjovsky and
               Vincent Dumoulin and
               Aaron C. Courville},
  title     = {Improved Training of Wasserstein GANs},
  journal   = {CoRR},
  volume    = {abs/1704.00028},
  year      = {2017},
  url       = {http://arxiv.org/abs/1704.00028},
  archivePrefix = {arXiv},
  eprint    = {1704.00028},
  timestamp = {Mon, 13 Aug 2018 16:47:43 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/GulrajaniAADC17.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```