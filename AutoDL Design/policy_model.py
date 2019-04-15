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
    PolicyModel definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import paddle.fluid as fluid
from parl.framework.model_base import Model
import distribute_generator


class LstmUnit(object):
    """
        implemetation of lstm unit
    """
    def __init__(self, input_size, hidden_size, num_layers=1,
                 init_scale=0.1):
        """
            init
        """
        self.weight_1_arr = []
        self.bias_1_arr = []
        for i in range(num_layers):
            weight_1 = fluid.layers.create_parameter(
                    [input_size + hidden_size, hidden_size * 4],
                    dtype="float32",
                    name="fc_weight1_" + str(i),
                    default_initializer=fluid.initializer.UniformInitializer(
                            low=-init_scale,
                            high=init_scale))
            input_size = hidden_size
            self.weight_1_arr.append(weight_1)
            bias_1 = fluid.layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            self.bias_1_arr.append(bias_1)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def lstm_step(self, inputs, hidden, cell):
        """
            lstm step
        """
        hidden_array = []
        cell_array = []
        for i in range(self.num_layers):
            hidden_temp = fluid.layers.slice(hidden, axes=[0], starts=[i],
                                             ends=[i + 1])
            hidden_temp = fluid.layers.reshape(hidden_temp,
                                               shape=[-1, self.hidden_size])
            hidden_array.append(hidden_temp)

            cell_temp = fluid.layers.slice(cell, axes=[0], starts=[i],
                                           ends=[i + 1])
            cell_temp = fluid.layers.reshape(cell_temp,
                                             shape=[-1, self.hidden_size])
            cell_array.append(cell_temp)

        last_hidden_array = []
        step_input = inputs
        for k in range(self.num_layers):
            pre_hidden = hidden_array[k]
            pre_cell = cell_array[k]
            weight = self.weight_1_arr[k]
            bias = self.bias_1_arr[k]

            nn = fluid.layers.concat([step_input, pre_hidden], 1)
            gate_input = fluid.layers.matmul(x=nn, y=weight)

            gate_input = fluid.layers.elementwise_add(gate_input, bias)
            i, j, f, o = fluid.layers.split(gate_input, num_or_sections=4,
                                            dim=-1)

            c = pre_cell * fluid.layers.sigmoid(f) + fluid.layers.sigmoid(i) \
                * fluid.layers.tanh(j)
            m = fluid.layers.tanh(c) * fluid.layers.sigmoid(o)

            hidden_array[k] = m
            cell_array[k] = c
            step_input = m

        last_hidden = fluid.layers.concat(hidden_array, axis=0)
        last_hidden = fluid.layers.reshape(last_hidden, shape=[
                self.num_layers, -1, self.hidden_size])

        last_cell = fluid.layers.concat(cell_array, axis=0)
        last_cell = fluid.layers.reshape(
                last_cell,
                shape=[self.num_layers, -1, self.hidden_size])
        return step_input, last_hidden, last_cell

    def __call__(self, inputs, hidden, cell):
        """
            lstm step call
        """
        return self.lstm_step(inputs, hidden, cell)


class PolicyModel(Model):
    """
        PolicyModel
    """
    def __init__(self, parser_args):
        """
            construct rnn net
        """
        self.parser_args = parser_args

    def policy(self, inputs):
        """
            policy function is used by `define_predict` in PolicyGradient
        """
        [tokens, softmax, adjvec, sigmoid] = self.build_rnn(inputs)
        return [tokens, softmax, adjvec, sigmoid]

    def build_rnn(self, inputs):
        """
            build rnn net
        """
        batch_size = self.parser_args.batch_size
        input_size = self.parser_args.input_size
        hidden_size = self.parser_args.hidden_size
        num_layers = self.parser_args.num_layers
        num_nodes = self.parser_args.num_nodes
        num_tokens = self.parser_args.num_tokens

        depth = max(num_nodes - 1, num_tokens)
        lstm_unit = LstmUnit(input_size, hidden_size, num_layers)

        def encode_token(inp):
            """
                encode token
            """
            token = fluid.layers.assign(inp)
            token.stop_gradient = True
            token = fluid.layers.one_hot(token, depth)
            return token

        def encode_adj(adj, step):
            """
                encode adj
            """
            adj = fluid.layers.cast(adj, dtype='float32')
            adj_pad = fluid.layers.pad(x=adj, paddings=[0, 0, 0, depth - step],
                                       pad_value=0.0)
            return adj_pad

        def decode_token(hidden):
            """
                decode token
            """
            initiallizer = fluid.initializer.TruncatedNormalInitializer(
                    scale=np.sqrt(2.0 / self.parser_args.hidden_size))
            param_attr = fluid.ParamAttr(initializer=initiallizer)
            logits = fluid.layers.fc(hidden, num_tokens, param_attr=param_attr)
            temper = 5.0
            tanh_c = 2.5
            logits = fluid.layers.tanh(logits / temper) * tanh_c
            token = distribute_generator.multinomial(logits,
                                                     [batch_size, 1], 1)
            return token, fluid.layers.unsqueeze(logits, axes=[1])

        def decode_adj(hidden, step):
            """
                decode adj
            """
            initiallizer = fluid.initializer.TruncatedNormalInitializer(
                    scale=np.sqrt(2.0 / self.parser_args.hidden_size))
            param_attr = fluid.ParamAttr(initializer=initiallizer)
            logits = fluid.layers.fc(hidden, step, param_attr=param_attr)
            temper = 5.0
            tanh_c = 2.5
            logits = fluid.layers.tanh(logits / temper) * tanh_c
            adj = distribute_generator.bernoulli(logits,
                                                 output_shape=logits.shape)
            return adj, logits

        tokens = []
        softmax = []
        adjvec = []
        sigmoid = []

        def rnn_block(hidden, last_hidden, last_cell):
            """
                rnn block
            """
            last_output, last_hidden, last_cell = lstm_unit(
                    hidden, last_hidden, last_cell)
            token, logits = decode_token(last_output)
            tokens.append(token)
            softmax.append(logits)

            for step in range(1, num_nodes):
                token_vec = encode_token(token)
                last_output, last_hidden, last_cell = lstm_unit(
                        token_vec, last_hidden, last_cell)
                adj, logits = decode_adj(last_output, step)
                adjvec.append(adj)
                sigmoid.append(logits)
                adj_vec = encode_adj(adj, step)
                last_output, last_hidden, last_cell = lstm_unit(
                        adj_vec, last_hidden, last_cell)
                token, logits = decode_token(last_output)
                tokens.append(token)
                softmax.append(logits)
            return token, last_hidden, last_cell
        init_hidden = fluid.layers.fill_constant(
                shape=[num_layers, batch_size, hidden_size],
                value=0.0, dtype='float32')
        init_cell = fluid.layers.fill_constant(
                shape=[num_layers, batch_size, hidden_size],
                value=0.0, dtype='float32')

        hidden = encode_adj(inputs, 1)
        token, last_hidden, last_cell = rnn_block(hidden, init_hidden,
                                                  init_cell)
        hidden = encode_token(token)
        token, last_hidden, last_cell = rnn_block(hidden, last_hidden,
                                                  last_cell)
        token_out = fluid.layers.concat(tokens, axis=1)
        softmax_out = fluid.layers.concat(softmax, axis=1)
        adjvec_out = fluid.layers.concat(adjvec, axis=1)
        sigmoid_out = fluid.layers.concat(sigmoid, axis=1)
        return [token_out, softmax_out, adjvec_out, sigmoid_out]
