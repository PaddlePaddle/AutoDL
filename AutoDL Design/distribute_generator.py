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
    Implementation of binomial and multinomial distribution
"""

import paddle.fluid as fluid
import functools
import numpy as np


def create_tmp_var(name, dtype, shape, program=None):
    """
        Create variable which is used to store the py_func result
    """
    if program is None:
        return fluid.default_main_program().current_block().create_var(
                name=fluid.unique_name.generate(name),
                dtype=dtype, shape=shape)
    else:
        return program.current_block().create_var(
                name=fluid.unique_name.generate(name),
                dtype=dtype, shape=shape)


def sigmoid(x):
    """
        Sigmoid
    """
    return (1 / (1 + np.exp(-x)))


def softmax(x):
    """
        Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def py_func_bernoulli(input):
    """
        Binormial python function definition
    """
    prob_array = sigmoid(np.array(input))
    sample = np.random.binomial(1, prob_array)
    return sample


def bernoulli(input_logits, output_shape, program=None):
    """
        Bernoulli
    """
    # the output_shape is the same as input_logits
    samples_var = create_tmp_var(name='binomial_result_var',
                                 dtype='float32', shape=output_shape,
                                 program=program)
    fluid.layers.py_func(func=py_func_bernoulli, x=input_logits,
                         out=samples_var, backward_func=None,
                         skip_vars_in_backward_input=None)
    return samples_var


def py_func_multinomial(logits, num_samples_var):
    """
        Multinomial python function definition
        Input:
            input: list of [logits_array, num_samples_int]
    """
    def generate(x, prob_array):
        """
            Sample multinomial
        """
        sample = np.random.multinomial(1, prob_array)
        ret = np.argmax(sample)
        return ret

    num_samples = int(np.array(num_samples_var)[0])
    logits_array = np.array(logits)
    if len(logits_array.shape) != 2:
        raise Exception("Shape must be rank 2 but is rank {} \
                        for 'multinomial/Multinomial' (op: 'Multinomial') \
                        with input shapes:{}".format(len(logits_array.shape),
                                                     logits_array.shape))
    ret = np.array([])
    for logits in logits_array:
        prob = softmax(logits)
        func = functools.partial(generate, prob_array=prob)
        sample = np.zeros(num_samples)
        sample = np.array(list(map(func, sample)))
        ret = np.append(ret, sample)
    ret = ret.reshape(-1, num_samples).astype("int32")
    return ret


def multinomial(input_logits, output_shape, num_samples, program=None):
    """
        Multinomial
        input_logits's dimension is [M * D]
        output_shape's dimension is [M * num_samples]
    """
    samples_var = create_tmp_var(name='multinomial_result_var',
                                 dtype='int32', shape=output_shape,
                                 program=program)
    num_samples_var = fluid.layers.fill_constant(shape=[1], value=num_samples,
                                                 dtype='int32')
    fluid.layers.py_func(func=py_func_multinomial,
                         x=[input_logits, num_samples_var],
                         out=samples_var, backward_func=None,
                         skip_vars_in_backward_input=None)
    return samples_var
