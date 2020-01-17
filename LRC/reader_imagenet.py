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
import cv2

__all__ = ['train10', 'test10']

train_image_size = 320
test_image_size = 256

CIFAR_MEAN = [0.485, 0.456, 0.406]
CIFAR_STD = [0.229, 0.224, 0.225]

def _parse_kv(r):
    """ parse kv data from sequence file for imagenet
    """
    import cPickle
    k, v = r
    obj = cPickle.loads(v)
    return obj['image'], obj['label']

def random_crop(img, size, scale=[0.08, 1.0], ratio=[3. / 4., 4. / 3.]):
    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    # PIL
    #bound = min((float(img.size[0]) / img.size[1]) / (w**2),
    #            (float(img.size[1]) / img.size[0]) / (h**2))
    # cv2
    bound = min((float(img.shape[1]) / img.shape[0]) / (w**2),
                (float(img.shape[0]) / img.shape[1]) / (h**2))
    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    # PIL
    #target_area = img.size[0] * img.size[1] * np.random.uniform(scale_min,
    #                                                            scale_max)
    #cv2
    target_area = img.shape[0] * img.shape[1] * np.random.uniform(scale_min,
            scale_max)

    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)

    # PIL
    #i = np.random.randint(0, img.size[0] - w + 1)
    #j = np.random.randint(0, img.size[1] - h + 1)

    #img = img.crop((i, j, i + w, j + h))
    #img = img.resize((size, size), Image.BILINEAR)
    # cv2
    i = np.random.randint(0, img.shape[0] - h + 1)
    j = np.random.randint(0, img.shape[1] - w + 1)
    img = img[i:i+h, j:j+w,:]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img

# PIL
"""
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
"""
# cv2
def crop_image(img, target_size, center=True):
    height, width = img.shape[:2]
    size = target_size
    if center == True:
        w_start = (width - size) / 2
        h_start = (height - size) / 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end,:]
    return img
    
# PIL
"""
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
"""
# cv2
def preprocess(img_path, is_training):
    
    img = cv2.imread(img_path)
    if is_training:
        # random resize crop
        img = random_crop(img, train_image_size)
        # random horizontal flip
        if np.random.randint(2):
            img = cv2.flip(img, 1)
    else:
        # resize
        img = cv2.resize(img, (test_image_size, test_image_size), interpolation=cv2.INTER_LINEAR)
        # center crop
        img = crop_image(img, train_image_size)

    img_float = img[:,:,::-1].astype('float32') / 255
    img = (img_float - CIFAR_MEAN) / CIFAR_STD
    img = np.transpose(img, (2, 0, 1))

    return img

def reader_creator_filepath(data_dir, sub_name, is_training):

    file_list = os.path.join(data_dir, sub_name)
    image_file = 'train' if is_training else 'val'
    dataset_path = os.path.join(data_dir, image_file)
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

def _reader_creator(data_dir, file_list,is_training):
    def multiprocess_reader():
        full_lines = [line.strip() for line in file_list]
        # NOTE:maybe do not need shuffle here!
        if is_training:
            np.random.shuffle(full_lines)
        for line in full_lines:
            img_path, label = line.split()
            img_path = os.path.join(data_dir, img_path)
            img = preprocess(img_path,is_training)
            yield img, int(label)
#    multiprocess_reader()
    return multiprocess_reader

def mul_reader_creator_filepath(data_dir, sub_name, is_training):

    file_list = os.path.join(data_dir, sub_name)
    image_file = 'train' if is_training else 'val'
    dataset_path = os.path.join(data_dir, image_file)
    
    with open(file_list,'r')as f_dir:
        lines = f_dir.readlines()
    
    num_workers = 16
  
    n = int(math.ceil(len(lines)/float(num_workers)))

#   global shuffle without image classification " pass seed " strategy
    if is_training:
        np.random.shuffle(lines)
    split_lists = [lines[i:i+n] for i in range(0,len(lines),n)]
    readers = []
    for item in split_lists:
        readers.append(
                _reader_creator(
                    dataset_path,
                    item,
                    'True'
                    )
                )
    return paddle.reader.multiprocess_reader(readers,False)



def train(args):
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """

   # return reader_creator_filepath(args.data, 'train.txt', True)
    return mul_reader_creator_filepath('./dataset/imagenet', 'train.txt', True)




def test(args):
    """
    CIFAR-10 test set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Test reader creator.
    :rtype: callable
    """
    return mul_reader_creator_filepath('./dataset/imagenet', 'val.txt', False)
 #   return reader_creator_filepath(args.data, 'val.txt', False)
