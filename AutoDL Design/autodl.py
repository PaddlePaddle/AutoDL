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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import subprocess
import paddle.fluid as fluid
from reinforce_policy_gradient import ReinforcePolicyGradient
from policy_model import PolicyModel
from autodl_agent import AutoDLAgent
import utils
import collections


class AutoDL(object):
    """
        AutoDL class
    """
    def __init__(self):
        """
            init
        """
        self.parse_args = self._init_parser()
        self.bl_decay = self.parse_args.bl_decay
        self.log_dir = self.parse_args.log_dir
        self.early_stop = self.parse_args.early_stop
        self.data_path = self.parse_args.data_path
        self.num_models = self.parse_args.num_models
        self.batch_size = self.parse_args.batch_size
        self.chunk_size= self.parse_args.chunk_size

        self._init_dir_path()
        self.model = PolicyModel(self.parse_args)
        algo_hyperparas = {'lr': self.parse_args.learning_rate}
        self.algorithm = ReinforcePolicyGradient(self.model,
                                                 hyperparas=algo_hyperparas)
        self.autodl_agent = AutoDLAgent(self.algorithm, self.parse_args)
        self.total_reward = 0

    def _init_dir_path(self):
        """
            init dir path
        """
        utils.prepare(self.log_dir)
        utils.prepare(self.log_dir, "actions")
        utils.prepare(self.log_dir, "rewards")
        utils.prepare(self.log_dir, "checkpoints")

    def _init_parser(self):
        """
            init parser
        """
        parser = argparse.ArgumentParser(description='AutoDL Parser',
                                         prog='AutoDL')
        parser.add_argument('-v', '--version', action='version',
                            version='%(prog)s 0.1')
        parser.add_argument('--num_nodes', dest="num_nodes", nargs="?",
                            type=int, const=10, default=10,
                            help="number of nodes")
        parser.add_argument('--num_tokens', dest="num_tokens", nargs="?",
                            type=int, const=10, default=10,
                            help="number of tokens")
        parser.add_argument('--learning_rate', dest="learning_rate", nargs="?",
                            type=float, default=1e-3,
                            help="learning rate")
        parser.add_argument('--batch_size', dest="batch_size", nargs="?",
                            type=int, const=10, default=10, help="batch size")
        parser.add_argument('--num_models', dest="num_models", nargs="?",
                            type=int, const=32000, default=32000,
                            help="maximum number of models sampled")
        parser.add_argument('--early_stop', dest="early_stop", nargs="?",
                            type=int, const=20, default=20, help="early stop")
        parser.add_argument('--log_dir', dest="log_dir", nargs="?", type=str,
                            const="./log", default="./log",
                            help="directory of log")
        parser.add_argument('--input_size', dest="input_size", nargs="?",
                            type=int, const=10, default=10, help="input size")
        parser.add_argument('--hidden_size', dest="hidden_size", nargs="?",
                            type=int, const=64, default=64, help="hidden size")
        parser.add_argument('--num_layers', dest="num_layers", nargs="?",
                            type=int, const=2, default=2, help="num layers")
        parser.add_argument('--bl_decay', dest="bl_decay", nargs="?",
                            type=float, const=0.9, default=0.9,
                            help="base line decay")
        # inception train config
        parser.add_argument('--data_path', dest="data_path", nargs="?",
                            type=str, default="./cifar/pickle-cifar-10",
                            help="path of data files")
        parser.add_argument('--chunk_size', dest="chunk_size", nargs="?",
                            type=int, const=100, default=100,
                            help="chunk size")
        parse_args = parser.parse_args()
        return parse_args

    def supervisor(self, mid):
        """
            execute cnn training
            sample cmd: python -u inception_train/train.py --mid=9 \
                    --early_stop=20 --data_path=./cifar/pickle-cifar-10
        """
        tokens, adjvec = utils.load_action(mid, self.log_dir)
        cmd = ("CUDA_VISIBLE_DEVICES=1 python -u inception_train/train.py \
               --mid=%d --early_stop=%d --logdir=%s --data_path=%s --chunk_size=%d") % \
              (mid, self.early_stop, self.log_dir, self.data_path, self.chunk_size)
        print("cmd:{}".format(cmd))
        while True:
            try:
                subprocess.check_call(cmd, shell=True)
                break
            except subprocess.CalledProcessError as e:
                print("[%s] training model #%d exits with exit code %d" %
                      (utils.stime(), mid, e.returncode), file=sys.stderr)
                return

    def simple_run(self):
        """
            simple run
        """
        print("Simple run target is 20")
        mid = 0
        shadow = 0
        is_first = True
        while mid <= self.num_models:
            actions_to, actions_ad = self.autodl_agent.sample()
            rewards = np.count_nonzero(actions_to == 1, axis=1).astype("int32")
            # moving average
            current_mean_reward = np.mean(rewards)
            if is_first:
                shadow = current_mean_reward
                is_first = False
            else:
                shadow = shadow * self.bl_decay \
                        + current_mean_reward * (1 - self.bl_decay)
            self.autodl_agent.learn((np.array(actions_to).astype("int32"),
                                    np.array(actions_ad).astype("int32")),
                                    rewards - shadow)

            if mid % 10 == 0:
                print('mid=%d, average rewards=%.3f' % (mid, np.mean(rewards)))
            mid += 1

    def run(self):
        """
            run
        """
        rewards = []
        mid = 0
        while mid <= self.num_models:
            actions_to, actions_ad = self.autodl_agent.sample()

            for action in zip(actions_to, actions_ad):
                utils.dump_action(mid, action, self.log_dir)
                self.supervisor(mid)
                current_reward = utils.load_reward(mid, self.log_dir)
                if not np.isnan(current_reward):
                    rewards.append(current_reward.item())
                mid += 1

            if len(rewards) % self.batch_size == 0:
                print("[%s] step = %d, average accuracy = %.3f" %
                      (utils.stime(), self.autodl_agent.global_step,
                       np.mean(rewards)))
                rewards_array = np.array(rewards).astype("float32")
                if self.total_reward == 0:
                    self.total_reward = rewards_array.mean()
                else:
                    self.total_reward = self.total_reward * self.bl_decay \
                            + (1 - self.bl_decay) * rewards_array.mean()
                rewards_array = rewards_array - self.total_reward
                self.autodl_agent.learn([actions_to.astype("int32"),
                                         actions_ad.astype("int32")],
                                        rewards_array ** 3)
                rewards = []
