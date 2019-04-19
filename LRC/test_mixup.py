#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
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
from learning_rate import cosine_decay
import numpy as np
import argparse
from model import NetworkCIFAR as Network
import reader_cifar as reader
import sys
import os
import time
import logging
import genotypes
import paddle.fluid as fluid
import shutil
import utils

parser = argparse.ArgumentParser("cifar")
# yapf: disable
parser.add_argument('--data', type=str, default='./dataset/cifar/cifar-10-batches-py/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=96, help='batch size')
parser.add_argument('--model_id', type=int, help='model id')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument( '--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument( '--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--pretrained_model', type=str, default='/model_0/final/', help='pretrained model to load')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--dump_path', type=str, default='prob_test_0.pkl', help='dump path')
# yapf: enable

args = parser.parse_args()

CIFAR_CLASSES = 10
dataset_train_size = 50000
image_size = 32
genotypes.DARTS = genotypes.MY_DARTS_list[args.model_id]
print(genotypes.DARTS)


def main():
    image_shape = [3, image_size, image_size]
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    logging.info("args = %s", args)
    genotype = eval("genotypes.%s" % args.arch)
    model = Network(args.init_channels, CIFAR_CLASSES, args.layers,
                    args.auxiliary, genotype)
    test(model, args, image_shape)


def build_program(args, is_train, model, im_shape):
    out = []
    py_reader = model.build_input(im_shape, is_train)
    prob, acc_1, acc_5 = model.test_model(py_reader, args.init_channels)
    out = [py_reader, prob, acc_1, acc_5]
    return out


def test(model, args, im_shape):

    test_py_reader, prob, acc_1, acc_5 = build_program(args, False, model,
                                                       im_shape)

    test_prog = fluid.default_main_program().clone(for_test=True)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # yapf: disable
    if args.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(args.pretrained_model, var.name))
        fluid.io.load_vars(exe, args.pretrained_model, predicate=if_exist)

    # yapf: enable

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 1
    compile_program = fluid.compiler.CompiledProgram(
        test_prog).with_data_parallel(exec_strategy=exec_strategy)
    test_reader = reader.test10(args)
    test_py_reader.decorate_paddle_reader(test_reader)

    test_fetch_list = [prob, acc_1, acc_5]
    prob = []
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    test_py_reader.start()
    test_start_time = time.time()
    step_id = 0
    try:
        while True:
            prev_test_start_time = test_start_time
            test_start_time = time.time()
            prob_v, acc_1_v, acc_5_v = exe.run(compile_program,
                                               test_prog,
                                               fetch_list=test_fetch_list)
            prob.append(list(np.array(prob_v)))
            top1.update(np.array(acc_1_v), np.array(prob_v).shape[0])
            top5.update(np.array(acc_5_v), np.array(prob_v).shape[0])
            if step_id % args.report_freq == 0:
                print('prob shape:', np.array(prob_v).shape)
                print("Step {}, acc_1 {}, acc_5 {}, time {}".format(
                    step_id,
                    np.array(acc_1_v),
                    np.array(acc_5_v), test_start_time - prev_test_start_time))
            step_id += 1
    except fluid.core.EOFException:
        test_py_reader.reset()
    np.concatenate(prob).dump(args.dump_path)
    print("top1 {0}, top5 {1}".format(top1.avg, top5.avg))


if __name__ == '__main__':
    main()
