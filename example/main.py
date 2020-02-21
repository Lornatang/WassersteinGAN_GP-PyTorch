# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""We introduce a new algorithm named WGAN, an alternative to traditional GAN training.
In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse,
and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore,
we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting
the deep connections to other distances between distributions.
"""
import argparse
import hashlib
import os
import random
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from wgangp_pytorch import Discriminator
from wgangp_pytorch import Generator
from wgangp_pytorch import calculate_gradient_penalty

parser = argparse.ArgumentParser(description="PyTorch Wasserstein GAN GP")
parser.add_argument("--dataroot", type=str, default="./data",
                    help="path to datasets")
parser.add_argument("name", type=str,
                    help="dataset name. Option: [mnist, fmnist]")
parser.add_argument("-g", "--generator-arch", metavar="STR",
                    help="generator model architecture")
parser.add_argument("-d", "--discriminator-arch", metavar="STR",
                    help="discriminator model architecture")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N",
                    help="number of data loading workers (default: 4)")
parser.add_argument("--epochs", default=200, type=int, metavar="N",
                    help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N",
                    help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size", default=64, type=int,
                    metavar="N",
                    help="mini-batch size (default: 64), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="learning rate. (Default=0.0002)")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_critic", type=int, default=5,
                    help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01,
                    help="lower and upper clip value for disc. weights")
parser.add_argument("-p", "--print-freq", default=100, type=int,
                    metavar="N", help="print frequency (default: 100)")
parser.add_argument("--netG", default="", type=str, metavar="PATH",
                    help="path to latest generator checkpoint (default: none)")
parser.add_argument("--netD", default="", type=str, metavar="PATH",
                    help="path to latest discriminator checkpoint (default: none)")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true",
                    help="evaluate model on validation set")
parser.add_argument("--pretrained", dest="pretrained", action="store_true",
                    help="use pre-trained model")
parser.add_argument("--world-size", default=-1, type=int,
                    help="number of nodes for distributed training")
parser.add_argument("--rank", default=-1, type=int,
                    help="node rank for distributed training")
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str,
                    help="url used to set up distributed training")
parser.add_argument("--dist-backend", default="nccl", type=str,
                    help="distributed backend")
parser.add_argument("--outf", default="./imgs",
                    help="folder to output images. (default=`./imgs`).")
parser.add_argument("--seed", default=None, type=int,
                    help="seed for initializing training.")
parser.add_argument("--gpu", default=0, type=int,
                    help="GPU id to use.")
parser.add_argument("--multiprocessing-distributed", action="store_true",
                    help="Use multi-processing distributed training to launch "
                         "N processes per node, which has N GPUs. This is the "
                         "fastest way to use PyTorch for either single node or "
                         "multi node data parallel training")

# Loss weight for gradient penalty
lambda_gp = 10


def main():
  args = parser.parse_args()

  try:
    os.makedirs(args.outf)
  except OSError:
    pass

  if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn("You have chosen to seed training. "
                  "This will turn on the CUDNN deterministic setting, "
                  "which can slow down your training considerably! "
                  "You may see unexpected behavior when restarting "
                  "from checkpoints.")

  if args.gpu is not None:
    warnings.warn("You have chosen a specific GPU. This will completely "
                  "disable data parallelism.")

  if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])

  args.distributed = args.world_size > 1 or args.multiprocessing_distributed

  ngpus_per_node = torch.cuda.device_count()
  if args.multiprocessing_distributed:
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
  else:
    # Simply call main_worker function
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
  args.gpu = gpu

  if args.gpu is not None:
    print(f"Use GPU: {args.gpu} for training!")

  if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
      args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
      # For multiprocessing distributed training, rank needs to be the
      # global rank among all the processes
      args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
  # create model
  if "g" in args.generator_arch:
    if args.pretrained:
      generator = Generator.from_pretrained(args.generator_arch)
      print(f"=> using pre-trained model `{args.generator_arch}`")
    else:
      print(f"=> creating model `{args.generator_arch}`")
      generator = Generator.from_name(args.generator_arch)
  else:
    warnings.warn("You have chosen a specific model architecture. This will "
                  "default use MNIST model architecture!")
    if args.pretrained:
      generator = Generator.from_pretrained("g-mnist")
      print(f"=> using pre-trained model `g-mnist`")
    else:
      print(f"=> creating model `g-mnist`")
      generator = Generator.from_name("g-mnist")

  if "d" in args.discriminator_arch:
    if args.pretrained:
      discriminator = Discriminator.from_pretrained(args.discriminator_arch)
      print(f"=> using pre-trained model `{args.discriminator_arch}`")
    else:
      print(f"=> creating model `{args.discriminator_arch}`")
      discriminator = Discriminator.from_name(args.discriminator_arch)
  else:
    warnings.warn("You have chosen a specific model architecture. This will "
                  "default use MNIST model architecture!")
    if args.pretrained:
      discriminator = Discriminator.from_pretrained("d-mnist")
      print(f"=> using pre-trained model `d-mnist`")
    else:
      print(f"=> creating model `d-mnist`")
      discriminator = Discriminator.from_name("d-mnist")

  if args.distributed:
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if args.gpu is not None:
      torch.cuda.set_device(args.gpu)
      generator.cuda(args.gpu)
      discriminator.cuda(args.gpu)
      # When using a single GPU per process and per
      # DistributedDataParallel, we need to divide the batch size
      # ourselves based on the total number of GPUs we have
      args.batch_size = int(args.batch_size / ngpus_per_node)
      args.workers = int(args.workers / ngpus_per_node)
      generator = torch.nn.parallel.DistributedDataParallel(generator, device_ids=[args.gpu])
      discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[args.gpu])
    else:
      generator.cuda()
      discriminator.cuda()
      # DistributedDataParallel will divide and allocate batch_size to all
      # available GPUs if device_ids are not set
      generator = torch.nn.parallel.DistributedDataParallel(generator)
      discriminator = torch.nn.parallel.DistributedDataParallel(discriminator)
  elif args.gpu is not None:
    torch.cuda.set_device(args.gpu)
    generator = generator.cuda(args.gpu)
    discriminator = discriminator.cuda(args.gpu)
  else:
    # DataParallel will divide and allocate batch_size to all available
    # GPUs
    generator = torch.nn.DataParallel(generator).cuda()
    discriminator = torch.nn.DataParallel(discriminator).cuda()

  optimizerG = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
  optimizerD = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

  # optionally resume from a checkpoint
  if args.netG:
    if os.path.isfile(args.netG):
      print(f"=> loading checkpoint `{args.netG}`")
      state_dict = torch.load(args.netG)
      generator.load_state_dict(state_dict)
      compress_model(state_dict, filename=args.netG, model_arch=args.generator_arch)
      print(f"=> loaded checkpoint `{args.netG}`")
    else:
      print(f"=> no checkpoint found at `{args.netG}`")
  if args.netD:
    if os.path.isfile(args.netD):
      print(f"=> loading checkpoint `{args.netD}`")
      state_dict = torch.load(args.netD)
      discriminator.load_state_dict(state_dict)
      compress_model(state_dict, filename=args.netD, model_arch=args.discriminator_arch)
      print(f"=> loaded checkpoint `{args.netD}`")
    else:
      print(f"=> no checkpoint found at `{args.netD}`")

  cudnn.benchmark = True

  if args.name == "mnist":
    dataset = datasets.MNIST(root=args.dataroot, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                             ]))
  elif args.name == "fmnist":
    dataset = datasets.FashionMNIST(root=args.dataroot, download=True,
                                    transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,)),
                                    ]))
  else:
    warnings.warn("You have chosen a specific dataset. This will "
                  "default use MNIST dataset!")
    dataset = datasets.MNIST(root=args.dataroot, download=True,
                             transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                             ]))

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                           shuffle=True, num_workers=int(args.workers))

  if args.evaluate:
    validate(generator, args)
    return

  for epoch in range(args.start_epoch, args.epochs):
    # train for one epoch
    train(dataloader, generator, discriminator, optimizerG, optimizerD, epoch, args)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                and args.rank % ngpus_per_node == 0):
      # do checkpointing
      torch.save(generator.state_dict(), f"{args.outf}/netG_epoch_{epoch}.pth")
      torch.save(discriminator.state_dict(), f"{args.outf}/netD_epoch_{epoch}.pth")


def train(dataloader, generator, discriminator, optimizerG, optimizerD, epoch, args):
  # switch to train mode
  generator.train()
  discriminator.train()

  for i, data in enumerate(dataloader, 0):
    # get batch size data
    real_images = data[0]
    if args.gpu is not None:
      real_images = real_images.cuda(args.gpu, non_blocking=True)
    batch_size = real_images.size(0)

    # Sample noise as generator input
    noise = torch.randn(batch_size, 100)
    if args.gpu is not None:
      noise = noise.cuda(args.gpu, non_blocking=True)

    ##############################################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ##############################################
    discriminator.zero_grad()

    # Train with real
    real_output = discriminator(real_images)
    errD_real = -torch.mean(real_output)
    D_x = real_output.mean().item()

    # Generate fake image batch with G
    fake_images = generator(noise)

    # Train with fake
    fake_output = discriminator(fake_images.detach())
    errD_fake = torch.mean(fake_output)
    D_G_z1 = fake_output.mean().item()

    # Gradient penalty
    gradient_penalty = calculate_gradient_penalty(discriminator, real_images.data, fake_images.data)

    # Add the gradients from the all-real and all-fake batches
    errD = errD_real + errD_fake + gradient_penalty * lambda_gp
    errD.backward()
    # Update D
    optimizerD.step()

    ##############################################
    # (2) Update G network: maximize log(D(G(z)))
    ##############################################
    generator.zero_grad()

    if i % args.n_critic == 0:
      fake_output = discriminator(fake_images)
      errG = -torch.mean(fake_output)
      errG.backward()
      D_G_z2 = fake_output.mean().item()
      # Update G
      optimizerG.step()

      print(f"[{epoch}/{args.epochs}][{i}/{len(dataloader)}] "
            f"Loss_D: {errD.item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"D_x: {D_x:.4f} "
            f"D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}")

    if i % args.print_freq == 0:
      vutils.save_image(real_images,
                        f"{args.outf}/real_samples.png",
                        normalize=True)
      fixed_noise = torch.randn(args.batch_size, 100)
      if args.gpu is not None:
        fixed_noise = fixed_noise.cuda(args.gpu, non_blocking=True)
      fake_images = generator(fixed_noise)
      vutils.save_image(fake_images.detach(),
                        f"{args.outf}/fake_samples_epoch_{epoch}.png",
                        normalize=True)


def validate(model, args):
  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    noise = torch.randn(args.batch_size, 100)
    if args.gpu is not None:
      noise = noise.cuda(args.gpu, non_blocking=True)
    fake = model(noise)
    vutils.save_image(fake.detach(), f"{args.outf}/fake.png", normalize=True)
  print("The fake image has been generated!")


def cal_file_md5(filename):
  """ Calculates the MD5 value of the file
  Args:
      filename: The path name of the file.

  Return:
      The MD5 value of the file.

  """
  with open(filename, "rb") as f:
    md5 = hashlib.md5()
    md5.update(f.read())
    hash_value = md5.hexdigest()
  return hash_value


def compress_model(state, filename, model_arch):
  model_folder = "../checkpoints"
  try:
    os.makedirs(model_folder)
  except OSError:
    pass

  new_filename = model_arch + "-" + cal_file_md5(filename)[:8] + ".pth"
  torch.save(state, os.path.join(model_folder, new_filename))


if __name__ == "__main__":
  main()
