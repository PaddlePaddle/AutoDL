# Copyright (c) 2019 PaddlePaddle Authors. All Rig hts Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on:
# --------------------------------------------------------
# DARTS
# Copyright (c) 2018, Hanxiao Liu.
# Licensed under the Apache License, Version 2.0;
# --------------------------------------------------------

from PIL import Image
from PIL import ImageOps
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle
import random
import utils
import paddle.fluid as fluid
import time
import os
import functools
import paddle.reader
import math

__all__ = ['train10', 'test10']

train_image_size = 224
test_image_size = 256

CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]

def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.size[0]) / img.size[1]) / (w**2),
                (float(img.size[1]) / img.size[0]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
                                                                scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    i = np.random.randint(0, img.size[0] - w + 1)
    j = np.random.randint(0, img.size[1] - h + 1)

    img = img.crop((i, j, i + w, j + h))
    img = img.resize((size, size), Image.BILINEAR)
    return img

def crop_image(img, target_size, center=True):
    width, height = img.size
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img.crop((w_start, h_start, w_end, h_end))
    return img

def preprocess(img_path, is_training):

    img = Image.open(img_path)
    
    if is_training:
        # ramdom resized crop
        img = random_crop(img, train_image_size)
        # random horizontal flip
        if np.random.randint(2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        # resize
        img = img.resize((test_image_size, test_image_size), Image.BILINEAR)
        # center crop
        img = crop_image(img, train_image_size)            

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype(np.float32)

    # per_image_standardization
    img_float = img / 255.0
    img = (img_float - CIFAR_MEAN) / CIFAR_STD
   
    img = np.transpose(img, (2, 0, 1))
    return img


def reader_creator_filepath(data_dir, sub_name, is_training):

    file_list = os.path.join(data_dir, sub_name)
    image_file = 'train' if is_training else 'val'
    dataset_path = os.path.join(data_dir, image_file)
    print(dataset_path)
    def reader():
        with open(file_list) as flist:
            lines = [line.strip() for line in flist]
            if is_training:
                np.random.shuffle(lines)
            for line in lines:
                img_path, label = line.split()
                #img_path = img_path.replace("JPEG", "jpeg")
                img_path_ = os.path.join(dataset_path, img_path)
                img = preprocess(img_path_, is_training)
                yield img, int(label)

    return reader


def train(args):
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """

    return reader_creator_filepath(args.data, 'debug.txt', True)


def test(args):
    """
    CIFAR-10 test set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator_filepath(args.data, 'val.txt', False)
