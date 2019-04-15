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
    Layers Definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import operator
import numpy as np
import paddle.fluid as fluid
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("weight_decay", 0.0001,
                   "weight decay")
flags.DEFINE_float("bn_decay", 0.9,
                   "batch norm decay")
flags.DEFINE_float("relu_leakiness", 0.1,
                   "relu leakiness")
flags.DEFINE_float("dropout_rate", 0.5,
                   "dropout rate")


def calc_padding(img_width, stride, dilation, filter_width):
    """
        calculate pixels to padding in order to keep input/output size same.
    """
    filter_width = dilation * (filter_width - 1) + 1
    if img_width % stride == 0:
        pad_along_width = max(filter_width - stride, 0)
    else:
        pad_along_width = max(filter_width - (img_width % stride), 0)
    return pad_along_width // 2, pad_along_width - pad_along_width // 2


def conv(inputs,
         filters,
         kernel,
         strides=None,
         dilation=None,
         num_groups=1,
         conv_param=None,
         name=None):
    """
        normal conv layer
    """
    if strides is None:
        strides = (1, 1)
    if dilation is None:
        dilation = (1, 1)
    if isinstance(kernel, (tuple, list)):
        n = operator.mul(*kernel) * inputs.shape[1]
    else:
        n = kernel * kernel * inputs.shape[1]

    # pad input
    padding = (0, 0, 0, 0) \
        + calc_padding(inputs.shape[2], strides[0], dilation[0], kernel[0]) \
        + calc_padding(inputs.shape[3], strides[1], dilation[1], kernel[1])
    if sum(padding) > 0:
        inputs = fluid.layers.pad(inputs, padding, 0)

    param_attr = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.TruncatedNormal(
            0.0, scale=np.sqrt(2.0 / n)),
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))

    return fluid.layers.conv2d(
        inputs,
        filters,
        kernel,
        stride=strides,
        padding=0,
        dilation=dilation,
        groups=num_groups,
        param_attr=param_attr if conv_param is None else conv_param,
        use_cudnn=False if num_groups == inputs.shape[1] == filters else True,
        name=name)


def sep(inputs, filters, kernel, strides=None, dilation=None, name=None):
    """
        Separable convolution layer
    """
    if strides is None:
        strides = (1, 1)
    if dilation is None:
        dilation = (1, 1)
    if isinstance(kernel, (tuple, list)):
        n_depth = operator.mul(*kernel)
    else:
        n_depth = kernel * kernel
    n_point = inputs.shape[1]

    if isinstance(strides, (tuple, list)):
        multiplier = strides[0]
    else:
        multiplier = strides

    depthwise_param = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.TruncatedNormal(
            0.0, scale=np.sqrt(2.0 / n_depth)),
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))

    pointwise_param = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.TruncatedNormal(
            0.0, scale=np.sqrt(2.0 / n_point)),
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))

    depthwise_conv = conv(
        inputs=inputs,
        kernel=kernel,
        filters=int(filters * multiplier),
        strides=strides,
        dilation=dilation,
        num_groups=int(filters * multiplier),
        conv_param=depthwise_param,
        name='depthwise_' + name)

    return conv(
        inputs=depthwise_conv,
        kernel=(1, 1),
        filters=int(filters * multiplier),
        strides=(1, 1),
        dilation=dilation,
        conv_param=pointwise_param,
        name='pointwise_' + name)


def maxpool(inputs, kernel, strides=None, name=None):
    """
        maxpool
    """
    if strides is None:
        strides = (1, 1)
    padding = (0, 0, 0, 0) \
        + calc_padding(inputs.shape[2], strides[0], 1, kernel[0]) \
        + calc_padding(inputs.shape[3], strides[1], 1, kernel[1])
    if sum(padding) > 0:
        inputs = fluid.layers.pad(inputs, padding, 0)

    return fluid.layers.pool2d(
            inputs, kernel, 'max', strides, pool_padding=0,
            ceil_mode=False, name=name)


def avgpool(inputs, kernel, strides=None, name=None):
    """
        avgpool
    """
    if strides is None:
        strides = (1, 1)
    padding_pixel = (0, 0, 0, 0)
    padding_pixel += calc_padding(inputs.shape[2], strides[0], 1, kernel[0])
    padding_pixel += calc_padding(inputs.shape[3], strides[1], 1, kernel[1])

    if padding_pixel[4] == padding_pixel[5] and padding_pixel[
            6] == padding_pixel[7]:
        # same padding pixel num on all sides.
        return fluid.layers.pool2d(
            inputs,
            kernel,
            'avg',
            strides,
            pool_padding=(padding_pixel[4], padding_pixel[6]),
            ceil_mode=False)
    elif padding_pixel[4] + 1 == padding_pixel[5] \
            and padding_pixel[6] + 1 == padding_pixel[7] \
            and strides == (1, 1):
        # different padding size: first pad then crop.
        x = fluid.layers.pool2d(
            inputs,
            kernel,
            'avg',
            strides,
            pool_padding=(padding_pixel[5], padding_pixel[7]),
            ceil_mode=False)
        x_shape = x.shape
        return fluid.layers.crop(
            x,
            shape=(-1, x_shape[1], x_shape[2] - 1, x_shape[3] - 1),
            offsets=(0, 0, 1, 1), name=name)
    else:
        # not support. use padding-zero and pool2d.
        print("Warning: use zero-padding in avgpool")
        outputs = fluid.layers.pad(inputs, padding_pixel, 0)
        return fluid.layers.pool2d(
                outputs, kernel, 'avg', strides, pool_padding=0,
                ceil_mode=False, name=name)


def global_avgpool(inputs, name=None):
    """
        global avgpool
    """
    return fluid.layers.reduce_mean(inputs, dim=[2, 3], name=name)


def fully_connected(inputs, units, name=None):
    """
        fully connected
    """
    n = inputs.shape[1]
    param_attr = fluid.param_attr.ParamAttr(
        initializer=fluid.initializer.TruncatedNormal(
            0.0, scale=np.sqrt(2.0 / n)),
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))

    return fluid.layers.fc(inputs,
                           units,
                           param_attr=param_attr)


def batch_norm(inputs, name=None):
    """
        batch norm
    """
    param_attr = fluid.param_attr.ParamAttr(
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))
    bias_attr = fluid.param_attr.ParamAttr(
        regularizer=fluid.regularizer.L2Decay(FLAGS.weight_decay))

    return fluid.layers.batch_norm(
        inputs, momentum=FLAGS.bn_decay, epsilon=0.001,
        param_attr=param_attr,
        bias_attr=bias_attr)


def relu(inputs, name=None):
    """
        relu
    """
    if FLAGS.relu_leakiness:
        return fluid.layers.leaky_relu(inputs, FLAGS.relu_leakiness, name=name)
    return fluid.layers.relu(inputs, name=name)


def bn_relu(inputs, name=None):
    """
        batch norm + rely layer
    """
    output = batch_norm(inputs)
    return fluid.layers.relu(output, name=name)


def dropout(inputs, name=None):
    """
        dropout layer
    """
    return fluid.layers.dropout(inputs, dropout_prob=FLAGS.dropout_rate,
                                dropout_implementation='upscale_in_train',
                                name=name)
