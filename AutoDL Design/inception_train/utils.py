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
"""
    Utils Definition
"""

import os
import pickle
import time
from absl import flags

FLAGS = flags.FLAGS


def stime():
    """
        stime
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def load_action(mid):
    """
        load action by mid
    """
    filename = os.path.join(FLAGS.logdir, "actions", "%d.pkl" % mid)
    return pickle.load(open(filename, "rb"))


def dump_action(mid, action):
    """
        dump action
    """
    filename = os.path.join(FLAGS.logdir, "actions", "%d.pkl" % mid)
    pickle.dump(action, open(filename, "wb"))


def dump_reward(mid, reward):
    """
        dump reward
    """
    filename = os.path.join(FLAGS.logdir, "rewards", "%d.pkl" % mid)
    pickle.dump(reward, open(filename, "wb"))


class AvgrageMeter(object):
    """
        AvgrageMeter for test
    """
    def __init__(self):
        """
            init
        """
        self.reset()

    def reset(self):
        """
            reset
        """
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """
            update
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
