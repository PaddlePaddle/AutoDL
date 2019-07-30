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
    Network Definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as cp
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
import paddle.fluid as fluid
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math
from paddle.fluid.initializer import init_on_cpu
from models import inception
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float("lr_max", 0.1,
                   "initial learning rate")
flags.DEFINE_float("lr_min", 0.0001,
                   "limiting learning rate")
flags.DEFINE_integer("batch_size", 128,
                     "batch size")
flags.DEFINE_integer("T_0", 200,
                     "number of epochs")
flags.DEFINE_integer("chunk_size", 100,
                     "chunk size")


class CIFARModel(object):
    """
        CIFARModel class
    """
    def __init__(self, tokens, adjvec, im_shape):
        """
            CIFARModel init
        """
        chunk_size = FLAGS.chunk_size
        self.batch_size = FLAGS.batch_size
        self.tokens = tokens
        self.adjvec = adjvec
        self.im_shape = im_shape
        max_step = chunk_size * 9 * FLAGS.T_0 // FLAGS.batch_size
        test_batch = chunk_size // FLAGS.batch_size

        def cosine_decay():
            """
                Applies cosine decay to the learning rate.
            """
            global_step = _decay_step_counter()
            with init_on_cpu():
                frac = (1 + ops.cos(global_step / max_step * math.pi)) / 2
            return FLAGS.lr_min + (FLAGS.lr_max - FLAGS.lr_min) * frac

        self.lr_strategy = cosine_decay

    def fn_model(self, py_reader):
        """
            fn model
        """
        self.image, self.label = fluid.layers.read_file(py_reader)
        self.loss, self.accuracy = inception.net(
                self.image, self.label, self.tokens, self.adjvec)
        return self.loss, self.accuracy

    def build_input(self, image_shape, is_train):
        """
            build_input
        """
        name = 'train_reader' if is_train else 'test_reader'
        py_reader = fluid.layers.py_reader(
            capacity=64,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True,
            name=name)
        return py_reader

    def build_program(self, main_prog, startup_prog, is_train):
        """
            build_program
        """
        out = []
        with fluid.program_guard(main_prog, startup_prog):
            py_reader = self.build_input(self.im_shape, is_train)
            if is_train:
                with fluid.unique_name.guard():
                    loss, accuracy = self.fn_model(py_reader)
                    optimizer = fluid.optimizer.Momentum(
                        learning_rate=self.lr_strategy(),
                        momentum=0.9,
                        use_nesterov=True)
                    optimizer.minimize(loss)
                    out = [py_reader, loss, accuracy]
            else:
                with fluid.unique_name.guard():
                    loss, accuracy = self.fn_model(py_reader)
                    out = [py_reader, loss, accuracy]
        return out
