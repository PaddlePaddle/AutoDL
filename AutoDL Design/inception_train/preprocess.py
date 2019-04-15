#!/usr/bin/env python
# -*- encoding:utf-8 -*-
# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
    Data preprocess
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from PIL import Image
from PIL import ImageOps
from PIL import ImageEnhance
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_boolean("random_flip_left_right", True,
                     "random flip left and right")
flags.DEFINE_boolean("random_flip_up_down", False,
                     "random flip up and down")
flags.DEFINE_boolean("random_brightness", False,
                     "randomly adjust brightness")
image_size = 32


def augmentation(sample, is_training):
    """
        augmentation
    """
    image_array = sample.reshape(3, image_size, image_size)
    rgb_array = np.transpose(image_array, (1, 2, 0))
    img = Image.fromarray(rgb_array, 'RGB')

    if is_training:
        # pad and crop
        img = ImageOps.expand(img, (4, 4, 4, 4), fill=0)  # pad to 40 * 40 * 3
        left_top = np.random.randint(9, size=2)  # rand 0 - 8
        img = img.crop((left_top[0], left_top[1], left_top[0] + image_size,
                        left_top[1] + image_size))

        if FLAGS.random_flip_left_right:
            if np.random.randint(2):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if FLAGS.random_flip_up_down:
            if np.random.randint(2):
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if FLAGS.random_brightness:
            delta = np.random.uniform(-0.3, 0.3) + 1.
            img = ImageEnhance.Brightness(img).enhance(delta)

    img = np.array(img).astype(np.float32)

    # per_image_standardization
    img_float = img / 255.0
    num_pixels = img_float.size
    img_mean = img_float.mean()
    img_std = img_float.std()
    scale = np.maximum(np.sqrt(num_pixels), img_std)
    img = (img_float - img_mean) / scale

    img = np.transpose(img, (2, 0, 1))
    return img
