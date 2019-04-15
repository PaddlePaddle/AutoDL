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
    Trainer Definition
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import reader
import sys
import os
import time
import paddle.fluid as fluid
import utils
import cPickle as cp
from absl import flags
from absl import app
from nn import CIFARModel

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path",
                    "./cifar/pickle-cifar-10",
                    "data path")
flags.DEFINE_string("logdir", "log",
                    "logging directory")
flags.DEFINE_integer("mid", 0,
                     "model id")
flags.DEFINE_integer("early_stop", 20,
                     "early stop")

image_size = 32


def main(_):
    """
        main
    """
    image_shape = [3, image_size, image_size]
    files = os.listdir(FLAGS.data_path)
    names = [each_item for each_item in files]
    np.random.shuffle(names)
    train_list = names[:9]
    test_list = names[-1]
    tokens, adjvec = utils.load_action(FLAGS.mid)

    model = CIFARModel(tokens, adjvec, image_shape)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    startup = fluid.Program()
    train_prog = fluid.Program()
    test_prog = fluid.Program()
    train_vars = model.build_program(train_prog, startup, True)
    test_vars = model.build_program(test_prog, startup, False)
    exe.run(startup)

    train_accuracy, epoch_id = train(model, FLAGS.early_stop,
                                     train_prog, train_vars, exe, train_list)
    if epoch_id < FLAGS.early_stop:
        utils.dump_reward(FLAGS.mid, train_accuracy)
    else:
        test_accuracy = test(model, test_prog, test_vars, exe, [test_list])
        utils.dump_reward(FLAGS.mid, test_accuracy)


def train(model, epoch_num, train_prog, train_vars, exe, data_list):
    """
        train
    """
    train_py_reader, loss_train, acc_train = train_vars
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True
    train_exe = fluid.ParallelExecutor(
        main_program=train_prog,
        use_cuda=True,
        loss_name=loss_train.name,
        exec_strategy=exec_strategy,
        build_strategy=build_strategy)

    train_reader = reader.train10(FLAGS.data_path, FLAGS.batch_size, data_list)
    train_py_reader.decorate_paddle_reader(train_reader)

    train_fetch_list = [loss_train, acc_train]
    epoch_start_time = time.time()
    for epoch_id in range(epoch_num):
        train_py_reader.start()
        epoch_end_time = time.time()
        if epoch_id > 0:
            print("Epoch {}, total time {}".format(epoch_id - 1, epoch_end_time
                                                   - epoch_start_time))
        epoch_start_time = epoch_end_time
        epoch_end_time
        start_time = time.time()
        step_id = 0
        try:
            while True:
                prev_start_time = start_time
                start_time = time.time()
                loss_v, acc_v = train_exe.run(
                    fetch_list=[v.name for v in train_fetch_list])
                if np.isnan(np.array(loss_v).mean()):
                    format_str = "[%s] jobs done, step = %d, loss = nan"
                    print(format_str % (utils.stime(), step_id))
                    return np.array(acc_v).mean(), epoch_id
                print("Epoch {}, Step {}, loss {}, acc {}, time {}".format(
                        epoch_id, step_id, np.array(loss_v).mean(),
                        np.array(acc_v).mean(), start_time - prev_start_time))
                step_id += 1
                sys.stdout.flush()
        except fluid.core.EOFException:
            train_py_reader.reset()
    return np.array(acc_v).mean(), epoch_id


def test(model, test_prog, test_vars, exe, data_list):
    """
        test
    """
    test_py_reader, loss_test, acc_test = test_vars
    test_prog = test_prog.clone(for_test=True)
    objs = utils.AvgrageMeter()

    test_reader = reader.test10(FLAGS.data_path, FLAGS.batch_size, data_list)
    test_py_reader.decorate_paddle_reader(test_reader)

    test_py_reader.start()
    test_fetch_list = [acc_test]
    test_start_time = time.time()
    step_id = 0
    try:
        while True:
            prev_test_start_time = test_start_time
            test_start_time = time.time()
            acc_v, = exe.run(
                test_prog, fetch_list=test_fetch_list)
            objs.update(np.array(acc_v), np.array(acc_v).shape[0])
            step_id += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    print("test acc {0}".format(objs.avg))
    return objs.avg


if __name__ == '__main__':
    app.run(main)
