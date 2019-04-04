# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter
import math
from paddle.fluid.initializer import init_on_cpu


def cosine_decay(learning_rate, num_epoch, steps_one_epoch):
    """Applies cosine decay to the learning rate.
    lr = 0.5 * (math.cos(epoch * (math.pi / 120)) + 1)
    """
    global_step = _decay_step_counter()

    with init_on_cpu():
        decayed_lr = learning_rate * \
                 (ops.cos(fluid.layers.floor(global_step / steps_one_epoch) \
                 * math.pi / num_epoch) + 1)/2
    return decayed_lr

    
def cosine_with_warmup_decay(learning_rate, lr_min, steps_one_epoch, 
                                  warmup_epochs, total_epoch, num_gpu):
    global_step = _decay_step_counter()
    epoch_idx = fluid.layers.floor(global_step / steps_one_epoch)
    
    lr = fluid.layers.create_global_var(
        shape=[1],
        value=0.0,
        dtype='float32',
        persistable=True,
        name="learning_rate")

    warmup_epoch_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(warmup_epochs), force_cpu=True)
    num_gpu_var = fluid.layers.fill_constant(
        shape=[1], dtype='float32', value=float(num_gpu), force_cpu=True)
    batch_idx = global_step - steps_one_epoch * epoch_idx 

    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(epoch_idx < warmup_epoch_var):
            epoch_ = (batch_idx + 1) / steps_one_epoch
            factor = 1 / num_gpu_var * (epoch_ * (num_gpu_var - 1) / warmup_epoch_var + 1)
            decayed_lr = learning_rate * factor * num_gpu_var
            fluid.layers.assign(decayed_lr, lr)
        epoch_ = (batch_idx + 1) / steps_one_epoch
        m = epoch_ / total_epoch        
        frac = (1 + ops.cos(math.pi * m)) / 2
        cosine_lr = (lr_min + (learning_rate - lr_min) * frac) * num_gpu_var
        with switch.default():
            fluid.layers.assign(cosine_lr, lr)

    return lr

