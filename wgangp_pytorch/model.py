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
import torch
import torch.nn as nn

from .utils import get_model_params
from .utils import load_pretrained_weights
from .utils import model_params


# Wassertein Generative model architecture from the One weird trick...
# <https://arxiv.org/abs/1701.07875>`_ paper.
class Generator(nn.Module):
  r""" An Generator model. Most easily loaded with the .from_name or
      .from_pretrained methods

  Args:
    global_params (namedtuple): A set of GlobalParams shared between blocks

  Examples:
      >>> import torch
      >>> from gan_pytorch import Generator
      >>> from gan_pytorch import Discriminator
      >>> generator = Generator.from_pretrained("g-mnist")
      >>> discriminator = Discriminator.from_pretrained("g-mnist")
      >>> generator.eval()
      >>> discriminator.eval()
      >>> noise = torch.randn(1, 100)
      >>> discriminator(generator(noise)).item()
  """

  def __init__(self, global_params=None):
    super(Generator, self).__init__()
    self.channels = global_params.channels
    self.image_size = global_params.image_size

    self.main = nn.Sequential(
      nn.Linear(global_params.noise, 128),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(128, 256),
      nn.BatchNorm1d(256, global_params.batch_norm_momentum),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(256, 512),
      nn.BatchNorm1d(512, global_params.batch_norm_momentum),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(512, 1024),
      nn.BatchNorm1d(1024, global_params.batch_norm_momentum),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(1024, self.channels * self.image_size * self.image_size),
      nn.Tanh()
    )

  def forward(self, x):
    r"""Defines the computation performed at every call.

    Args:
      x (tensor): input tensor into the calculation.

    Returns:
      A four-dimensional vector (NCHW).
    """
    x = self.main(x)
    x = x.reshape(x.size(0), self.channels, self.image_size, self.image_size)
    return x

  @classmethod
  def from_name(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    global_params = get_model_params(model_name)
    return cls(global_params)

  @classmethod
  def from_pretrained(cls, model_name):
    model = cls.from_name(model_name, )
    load_pretrained_weights(model, model_name)
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, res = model_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. """
    valid_list = ["mnist", "fmnist"]
    valid_models = ["g-" + str(i) for i in valid_list]
    if model_name not in valid_models:
      raise ValueError("model_name should be one of: " + ", ".join(valid_models))


class Discriminator(nn.Module):
  r""" An Discriminator model. Most easily loaded with the .from_name or
      .from_pretrained methods

  Args:
    global_params (namedtuple): A set of GlobalParams shared between blocks

  Examples:
    >>> import torch
    >>> from gan_pytorch import Discriminator
    >>> discriminator = Discriminator.from_pretrained("d-mnist")
    >>> discriminator.eval()
    >>> noise = torch.randn(1, 784)
    >>> discriminator(noise).item()
  """

  def __init__(self, global_params=None):
    super(Discriminator, self).__init__()

    self.main = nn.Sequential(
      nn.Linear(global_params.channels * global_params.image_size * global_params.image_size, 512),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(512, 256),
      nn.LeakyReLU(global_params.negative_slope, inplace=True),

      nn.Linear(256, 1),
    )

  def forward(self, x):
    r""" Defines the computation performed at every call.

    Args:
      x (tensor): input tensor into the calculation.

    Returns:
      A four-dimensional vector (NCHW).
    """
    x = torch.flatten(x, 1)
    x = self.main(x)
    return x

  @classmethod
  def from_name(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    global_params = get_model_params(model_name)
    return cls(global_params)

  @classmethod
  def from_pretrained(cls, model_name):
    model = cls.from_name(model_name)
    load_pretrained_weights(model, model_name)
    return model

  @classmethod
  def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, res = model_params(model_name)
    return res

  @classmethod
  def _check_model_name_is_valid(cls, model_name):
    """ Validates model name. """
    valid_list = ["mnist", "fmnist"]
    valid_models = ["d-" + str(i) for i in valid_list]
    if model_name not in valid_models:
      raise ValueError("model_name should be one of: " + ", ".join(valid_models))
