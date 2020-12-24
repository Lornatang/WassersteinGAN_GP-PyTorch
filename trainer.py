# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
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
import logging
import os

import torch.utils.data
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm

import wgangp_pytorch.models as models
from wgangp_pytorch.models import discriminator
from wgangp_pytorch.utils import calculate_gradient_penalty
from wgangp_pytorch.utils import init_torch_seeds
from wgangp_pytorch.utils import select_device
from wgangp_pytorch.utils import weights_init

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


class Trainer(object):
    def __init__(self, args):
        self.args = args

        # Set random initialization seed, easy to reproduce.
        init_torch_seeds(args.manualSeed)

        logger.info("Load training dataset")
        # Selection of appropriate treatment equipment.
        if args.dataset == "mnist":
            dataset = torchvision.datasets.MNIST(root=args.dataroot, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(args.image_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))
        elif args.dataset == "fashion-mnist":
            dataset = torchvision.datasets.FashionMNIST(root=args.dataroot, download=True,
                                                        transform=transforms.Compose([
                                                            transforms.Resize(args.image_size),
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.5,), (0.5,))
                                                        ]))
        elif args.dataset == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root=args.dataroot, download=True,
                                                   transform=transforms.Compose([
                                                       transforms.Resize(args.image_size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ]))

        else:
            dataset = torchvision.datasets.MNIST(root=args.dataroot, download=True,
                                                 transform=transforms.Compose([
                                                     transforms.Resize(args.image_size),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.5,), (0.5,))
                                                 ]))

        self.dataloader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=args.batch_size,
                                                      pin_memory=True,
                                                      num_workers=int(args.workers))

        logger.info(f"Train Dataset information:\n"
                    f"\tTrain Dataset dir is `{os.getcwd()}/{args.dataroot}`\n"
                    f"\tBatch size is {args.batch_size}\n"
                    f"\tWorkers is {int(args.workers)}\n"
                    f"\tLoad dataset to CUDA")

        # Construct network architecture model of generator and discriminator.
        self.device = select_device(args.device, batch_size=1)
        if args.pretrained:
            logger.info(f"Using pre-trained model `{args.arch}`")
            self.generator = models.__dict__[args.arch](pretrained=True).to(self.device)
        else:
            logger.info(f"Creating model `{args.arch}`")
            self.generator = models.__dict__[args.arch]().to(self.device)
        logger.info(f"Creating discriminator model")
        self.discriminator = discriminator(image_size=args.image_size, channels=args.channels).to(self.device)

        self.generator = self.generator.apply(weights_init)
        self.discriminator = self.discriminator.apply(weights_init)

        # Parameters of pre training model.
        self.epochs = int(int(args.iters) // len(self.dataloader))
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=args.lr, betas=(0.5, 0.999))

        logger.info(f"Model training parameters:\n"
                    f"\tIters is {int(args.iters)}\n"
                    f"\tEpoch is {int(self.epochs)}\n"
                    f"\tOptimizer Adam\n"
                    f"\tBetas is (0.5, 0.999)\n"
                    f"\tLearning rate {args.lr}")

    def run(self):
        args = self.args

        self.discriminator.train()
        self.generator.train()

        # Load pre training model.
        if args.netD != "":
            self.discriminator.load_state_dict(torch.load(args.netD))
        if args.netG != "":
            self.generator.load_state_dict(torch.load(args.netG))

        # Start train PSNR model.
        logger.info(f"Training for {self.epochs} epochs")

        fixed_noise = torch.randn(args.batch_size, 100, device=self.device)

        for epoch in range(args.start_epoch, self.epochs):
            progress_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
            for i, data in progress_bar:
                real_images = torch.autograd.Variable(data[0].type(torch.Tensor), requires_grad=True)
                real_images = real_images.to(self.device)
                batch_size = real_images.size(0)

                # Sample noise as generator input
                noise = torch.randn(batch_size, 100, device=self.device)

                ##############################################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ##############################################
                # Set discriminator gradients to zero.
                self.discriminator.zero_grad()

                # Train with real
                real_output = self.discriminator(real_images)
                errD_real = -torch.mean(real_output)
                D_x = real_output.mean().item()

                # Generate fake image batch with G
                fake_images = self.generator(noise)

                # Train with fake
                fake_output = self.discriminator(fake_images)
                errD_fake = torch.mean(fake_output)
                D_G_z1 = fake_output.mean().item()

                # Calculate W-div gradient penalty
                gradient_penalty = calculate_gradient_penalty(self.discriminator, real_images, fake_images, self.device)

                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake + gradient_penalty * 10
                errD.backward()
                # Update D
                self.optimizer_d.step()

                # Train the generator every n_critic iterations
                if (i + 1) % args.n_critic == 0:
                    ##############################################
                    # (2) Update G network: maximize log(D(G(z)))
                    ##############################################
                    # Set generator gradients to zero
                    self.generator.zero_grad()
                    # Generate fake image batch with G
                    fake_images = self.generator(noise)
                    fake_output = self.discriminator(fake_images)
                    errG = -torch.mean(fake_output)
                    D_G_z2 = fake_output.mean().item()
                    errG.backward()
                    self.optimizer_g.step()

                    progress_bar.set_description(f"[{epoch + 1}/{self.epochs}][{i + 1}/{len(self.dataloader)}] "
                                                 f"Loss_D: {errD.item():.6f} Loss_G: {errG.item():.6f} "
                                                 f"D(x): {D_x:.6f} D(G(z)): {D_G_z1:.6f}/{D_G_z2:.6f}")

                # The image is saved every 1 epoch.
                if (i + 1) % args.save_freq == 0:
                    vutils.save_image(real_images, os.path.join("output", "real_samples.bmp"))
                    fake_images = self.generator(fixed_noise)
                    vutils.save_image(fake_images.detach(), os.path.join("output", f"fake_samples_{epoch + 1}.bmp"))

            # do checkpointing
            torch.save(self.generator.state_dict(), f"weights/netG_epoch_{epoch + 1}.pth")
            torch.save(self.discriminator.state_dict(), f"weights/netD_epoch_{epoch + 1}.pth")
