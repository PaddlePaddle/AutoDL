#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
    Generate pkl files from cifar10
"""
import os
import cPickle as pickle
import random
import numpy as np
import sys
import argparse


def init_parser():
    """
        init_parser
    """
    parser = argparse.ArgumentParser(description='Data generator')
    parser.add_argument('--chunk_size', dest="chunk_size", nargs="?",
                        type=int, default=100,
                        help="size of chunk")
    parser.add_argument('--input_dir', dest="input_dir", nargs="?",
                        type=str, default='./cifar-10-batches-py',
                        help="path of input")
    parser.add_argument('--output_dir', dest="output_dir", nargs="?",
                        type=str, default='./pickle-cifar-10',
                        help="path of output")
    parse_args, unknown_flags = parser.parse_known_args()
    return parse_args


def get_file_names(input_dir):
    """
        get all file names located in dir_path
    """
    sub_name = 'data_batch'
    files = os.listdir(input_dir)
    names = [each_item for each_item in files if sub_name in each_item]
    return names


def check_output_dir(output_dir):
    """
        check exist of output dir
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


def get_datasets(input_dir, chunk_size):
    """
        get image datasets
        chunk_size is the number of each class
    """
    total_size = chunk_size * 10
    names = get_file_names(parse_args.input_dir)
    img_count = 0
    datasets = []
    class_map = {i: 0 for i in range(10)}
    for name in names:
        print("Reading file " + name)
        batch = pickle.load(open(input_dir + "/" + name, 'rb'))
        data = batch['data']
        labels = batch.get('labels', batch.get('fine_labels', None))
        assert labels is not None
        data_tuples = zip(data, labels)
        for data in data_tuples:
            if class_map[data[1]] < chunk_size:
                datasets.append(data)
                class_map[data[1]] += 1
                img_count += 1
                if img_count >= total_size:
                    random.shuffle(datasets)
                    for k, v in class_map.items():
                        print("label:{} count:{}".format(k, v))
                    return np.array(datasets)
    random.shuffle(datasets)
    return np.array(datasets)


def dump_pkl(datasets, output_dir):
    """
        dump_pkl
    """
    chunk_size = parse_args.chunk_size
    for i in range(10):
        sub_dataset = datasets[i * chunk_size:(i + 1) * chunk_size, :]
        sub_dataset.dump(output_dir + "/" + 'data_batch_' + str(i) + '.pkl')


if __name__ == "__main__":
    parse_args = init_parser()
    check_output_dir(parse_args.output_dir)
    datasets = get_datasets(parse_args.input_dir, parse_args.chunk_size)
    dump_pkl(datasets, parse_args.output_dir)
