#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
"""
    CIFAR-10 dataset.
    This module will download dataset from
    https://www.cs.toronto.edu/~kriz/cifar.html and parse train/test set into
    paddle reader creators.
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes,
    with 6000 images per class. There are 50000 training images
    and 10000 test images.
"""

import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle
import random
import utils
import paddle.fluid as fluid
import os
from preprocess import augmentation


def reader_creator_filepath(filename, sub_name, is_training,
                            batch_size, data_list):
    """
        reader creator
    """
    dataset = []
    for name in data_list:
        print("Reading file " + name)
        file_path = os.path.join(filename, name)
        batch_data = pickle.load(open(file_path))
        dataset.append(batch_data)
    datasets = np.concatenate(dataset)
    if is_training:
        np.random.shuffle(dataset)

    def read_batch(datasets, is_training):
        """
            read batch
        """
        for sample, label in datasets:
            im = augmentation(sample, is_training)
            yield im, [int(label)]

    def reader():
        """
            get reader
        """
        batch_data = []
        batch_label = []
        for data, label in read_batch(datasets, is_training):
            batch_data.append(data)
            batch_label.append(label)
            if len(batch_data) == batch_size:
                batch_data = np.array(batch_data, dtype='float32')
                batch_label = np.array(batch_label, dtype='int64')
                batch_out = [[batch_data, batch_label]]
                yield batch_out
                batch_data = []
                batch_label = []
        if len(batch_data) != 0:
            batch_data = np.array(batch_data, dtype='float32')
            batch_label = np.array(batch_label, dtype='int64')
            batch_out = [[batch_data, batch_label]]
            yield batch_out
            batch_data = []
            batch_label = []
    return reader


def train10(data, batch_size, data_list):
    """
    CIFAR-10 training set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Training reader creator
    :rtype: callable
    """
    return reader_creator_filepath(data, 'data_batch', True,
                                   batch_size, data_list)


def test10(data, batch_size, data_list):
    """
    CIFAR-10 test set creator.
    It returns a reader creator, each sample in the reader is image pixels in
    [0, 1] and label in [0, 9].
    :return: Test reader creator.
    :rtype: callable
    """
    return reader_creator_filepath(data, 'test_batch', False,
                                   batch_size, data_list)
