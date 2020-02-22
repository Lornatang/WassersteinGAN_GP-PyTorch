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
import os
import time

import torch
import torchvision.utils as vutils
from django.shortcuts import render
from rest_framework.views import APIView

from wgangp_pytorch import Generator

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Generator.from_pretrained("g-fmnist")
model.to(device)
# switch to evaluate mode
model.eval()


def index(request):
  """Get the image based on the base64 encoding or url address
  and do the pencil style conversion

  Args:
    request: Post request in url.
      - image_code: 64-bit encoding of images.
      - url:        The URL of the image.

  Returns:
    Base64 bit encoding of the image.

  Notes:
    Later versions will not return an image's address,
    but instead a base64-bit encoded address
  """

  return render(request, "index.html")


class FMNIST(APIView):
  @staticmethod
  def get(request):
    """ Get the image based on the base64 encoding or url address

    Args:
      request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.

    Returns:
      Base64 bit encoding of the image.

    Notes:
      Later versions will not return an image's address,
      but instead a base64-bit encoded address
    """

    context = {
      "status_code": 20000,
      "message": None,
      "filename": None}
    return render(request, "fmnist.html", context)

  @staticmethod
  def post(request):
    """ Get the image based on the base64 encoding or url address

    Args:
      request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.

    Returns:
      Base64 bit encoding of the image.

    Notes:
      Later versions will not return an image's address,
      but instead a base64-bit encoded address
    """

    base_path = "static/fmnist"
    filename = str(time.time()) + ".png"

    try:
      os.makedirs(base_path)
    except OSError:
      pass

    with torch.no_grad():
      noise = torch.randn(64, 100, device=device)
      fake = model(noise)
      vutils.save_image(fake.detach().cpu(), os.path.join(base_path, filename), normalize=True)

    context = {
      "status_code": 20000,
      "message": "The fake image has been generated!",
      "filename": filename}
    return render(request, "fmnist.html", context)