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

import os
import time
import pickle


def stime():
    """
        stime
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def prepare(log_dir, category=""):
    """
        prepare directory
    """
    subdir = os.path.join(log_dir, category)
    if not os.path.exists(subdir):
        os.mkdir(subdir)


def dump_action(mid, action, log_dir):
    """
        dump action
    """
    filename = os.path.join(log_dir, "actions", "%d.pkl" % mid)
    pickle.dump(action, open(filename, "wb"))


def load_action(mid, log_dir):
    """
        load action
    """
    filename = os.path.join(log_dir, "actions", "%d.pkl" % mid)
    return pickle.load(open(filename, "rb"))


def dump_reward(mid, reward, log_dir):
    """
        dump reward
    """
    filename = os.path.join(log_dir, "rewards", "%d.pkl" % mid)
    pickle.dump(reward, open(filename, "wb"))


def load_reward(mid, log_dir):
    """
        load reward
    """
    filename = os.path.join(log_dir, "rewards", "%d.pkl" % mid)
    return pickle.load(open(filename, "rb"))
