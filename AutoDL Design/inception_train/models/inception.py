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
    Inception Definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from absl import flags
import numpy as np
import models.layers as layers
import models.ops as _ops

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_stages", 3, "number of stages")
flags.DEFINE_integer("num_cells", 3, "number of cells per stage")
flags.DEFINE_integer("width", 64, "network width")
flags.DEFINE_integer("ratio", 4, "compression ratio")

num_classes = 10

ops = [
    _ops.conv_1x1,
    _ops.conv_3x3,
    _ops.conv_5x5,
    _ops.dilated_3x3,
    _ops.conv_1x3_3x1,
    _ops.conv_1x5_5x1,
    _ops.maxpool_3x3,
    _ops.maxpool_5x5,
    _ops.avgpool_3x3,
    _ops.avgpool_5x5,
]


def net(inputs, output, tokens, adjvec):
    """
        create net
    """
    num_nodes = len(tokens) // 2

    def slice(vec):
        """
            slice vec
        """
        mat = np.zeros([num_nodes, num_nodes])

        def pos(x):
            """
                pos
            """
            return x * (x - 1) // 2
        for i in range(1, num_nodes):
            mat[0:i, i] = vec[pos(i):pos(i + 1)]
        return mat

    normal_to, reduce_to = np.split(tokens, 2)
    normal_ad, reduce_ad = map(slice, np.split(adjvec, 2))

    x = layers.conv(inputs, FLAGS.width, (3, 3))

    c = 1
    for _ in range(FLAGS.num_cells):
        x = cell(x, normal_to, normal_ad)
        c += 1
    for _ in range(1, FLAGS.num_stages):
        x = cell(x, reduce_to, reduce_ad, downsample=True)
        c += 1
        for _ in range(1, FLAGS.num_cells):
            x = cell(x, normal_to, normal_ad)
            c += 1

    x = layers.bn_relu(x)
    x = layers.global_avgpool(x)
    x = layers.dropout(x)
    logits = layers.fully_connected(x, num_classes)
    x = fluid.layers.softmax_with_cross_entropy(logits, output,
                                                numeric_stable_mode=True)
    loss = fluid.layers.reduce_mean(x)
    accuracy = fluid.layers.accuracy(input=logits, label=output)
    return loss, accuracy


def cell(inputs, tokens, adjmat, downsample=False, name=None):
    """
        cell
    """
    filters = inputs.shape[1]
    d = filters // FLAGS.ratio

    num_nodes, tensors = len(adjmat), []
    for n in range(num_nodes):
        func = ops[tokens[n]]
        idx, = np.nonzero(adjmat[:, n])
        if len(idx) == 0:
            x = layers.bn_relu(inputs)
            x = layers.conv(x, d, (1, 1))
            x = layers.bn_relu(x)
            x = func(x, downsample)
        else:
            x = fluid.layers.sums([tensors[i] for i in idx])
            x = layers.bn_relu(x)
            x = func(x)
        tensors.append(x)

    free_ends, = np.where(~adjmat.any(axis=1))
    tensors = [tensors[i] for i in free_ends]
    filters = filters * 2 if downsample else filters
    x = fluid.layers.concat(tensors, axis=1)
    x = layers.conv(x, filters, (1, 1))
    return x
