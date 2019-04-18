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
    AutoDL Agent Definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from parl.framework.agent_base import Agent


class AutoDLAgent(Agent):
    """
        AutoDLAgent
    """
    def __init__(self, algorithm, parse_args):
        """
            init
        """
        self.global_step = 0
        self.parse_args = parse_args
        self.num_nodes = self.parse_args.num_nodes
        self.batch_size = self.parse_args.batch_size
        super(AutoDLAgent, self).__init__(algorithm)
        self.inputs_data = np.zeros([self.batch_size,
                                     1]).astype('int32')

    def build_program(self):
        """
            build program
        """
        self.predict_program = fluid.Program()
        self.train_program = fluid.Program()
        with fluid.program_guard(self.predict_program):
            self.predict_inputs = layers.data(
                    name='input',
                    append_batch_size=False,
                    shape=[self.batch_size, 1],
                    dtype='int32')
            self.predict_tokens, self.predict_adjvec = self.alg.define_predict(
                    self.predict_inputs)

        with fluid.program_guard(self.train_program):
            self.train_inputs = layers.data(
                    name='input',
                    append_batch_size=False,
                    shape=[self.batch_size, 1],
                    dtype='int32')
            self.actions_to = layers.data(
                    name='actions_to',
                    append_batch_size=False,
                    shape=[self.batch_size,
                           self.num_nodes * 2],
                    dtype='int32')
            self.actions_ad = layers.data(
                    name='actions_ad',
                    append_batch_size=False,
                    shape=[self.batch_size,
                           self.num_nodes * (self.num_nodes - 1)],
                    dtype='int32')
            self.rewards = layers.data(
                name='rewards',
                append_batch_size=False,
                shape=[self.batch_size],
                dtype='float32')
            self.cost = self.alg.define_learn(
                    obs=self.train_inputs, reward=self.rewards,
                    action=[self.actions_to, self.actions_ad])

    def sample(self):
        """
            sample
        """
        feed_dict = {'input': self.inputs_data}
        [actions_to, actions_ad] = self.fluid_executor.run(
                self.predict_program, feed=feed_dict,
                fetch_list=[self.predict_tokens, self.predict_adjvec])
        return actions_to, actions_ad

    def learn(self, actions, reward):
        """
            learn
        """
        (actions_to, actions_ad) = actions
        feed_dict = {'input': self.inputs_data, 'actions_to': actions_to,
                     'actions_ad': actions_ad, 'rewards': reward}
        cost = self.fluid_executor.run(
                self.train_program, feed=feed_dict, fetch_list=[self.cost])[0]
        self.global_step += 1
        return cost
