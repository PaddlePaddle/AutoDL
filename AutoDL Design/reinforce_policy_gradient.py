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
    AutoDL definition
"""
import paddle.fluid as fluid
from parl.framework.algorithm_base import Algorithm
import paddle.fluid.layers as layers
import os
import sys


class ReinforcePolicyGradient(Algorithm):
    """
        Implement REINFORCE policy gradient for autoDL
    """
    def __init__(self, model, hyperparas):
        """
        """
        Algorithm.__init__(self, model, hyperparas)
        self.model = model
        self.lr = hyperparas['lr']

    def define_predict(self, obs):
        """
            use policy model self.model to predict the action probability
            obs is `inputs`
        """
        with fluid.unique_name.guard():
            [tokens, softmax, adjvec, sigmoid] = self.model.policy(obs)
            return tokens, adjvec

    def define_learn(self, obs, action, reward):
        """
            update policy model self.model with policy gradient algorithm
            obs is `inputs`
        """
        tokens = action[0]
        adjvec = action[1]
        with fluid.unique_name.guard():
            [_, softmax, _, sigmoid] = self.model.policy(obs)
            reshape_softmax = layers.reshape(
                    softmax,
                    [-1, self.model.parser_args.num_tokens])
            reshape_tokens = layers.reshape(tokens, [-1, 1])
            reshape_tokens.stop_gradient = True
            raw_neglogp_to = layers.softmax_with_cross_entropy(
                    soft_label=False,
                    logits=reshape_softmax,
                    label=fluid.layers.cast(x=reshape_tokens, dtype="int64"))

            action_to_shape_sec = self.model.parser_args.num_nodes * 2
            neglogp_to = layers.reshape(fluid.layers.cast(
                                            raw_neglogp_to, dtype="float32"),
                                        [-1, action_to_shape_sec])

            adjvec = layers.cast(x=adjvec, dtype='float32')
            neglogp_ad = layers.sigmoid_cross_entropy_with_logits(
                    x=sigmoid, label=adjvec)

            neglogp = layers.elementwise_add(
                    x=layers.reduce_sum(neglogp_to, dim=1),
                    y=layers.reduce_sum(neglogp_ad, dim=1))
            reward = layers.cast(reward, dtype="float32")
            cost = layers.reduce_mean(
                    fluid.layers.elementwise_mul(x=neglogp, y=reward))
            optimizer = fluid.optimizer.Adam(learning_rate=self.lr)
            train_op = optimizer.minimize(cost)
            return cost
