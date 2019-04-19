# LRC 局部Rademachar复杂度正则化
为了在深度神经网络中提升泛化能力，正则化的选择十分重要也具有挑战性。本目录包括了一种基于局部rademacher复杂度的新型正则（LRC）的图像分类模型。十分感谢[DARTS](https://arxiv.org/abs/1806.09055)模型对本研究提供的帮助。该模型将LRC正则和DARTS网络相结合，在CIFAR-10数据集中得到了98.01%的准确率。代码和文章一同发布
> [An Empirical Study on Regularization of Deep Neural Networks by Local Rademacher Complexity](https://arxiv.org/abs/1902.00873)\
> Yingzhen Yang, Xingjian Li, Jun Huan.\
> _arXiv:1902.00873_.

---
# 内容

- [算法简介](#算法简介)
- [安装](#安装)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型测试](#模型测试)
- [实验结果](#实验结果)
- [引用](#引用)

## 算法简介

局部拉德马赫复杂度方法借鉴了已有的局部拉德马赫复杂度方法，仅考虑在经验损失函数的极小值点附近的一个球内的拉德马赫复杂度。采用最近的拉德马赫复杂度的估计方法，对折页损失函数 (Hinge Loss) 和交叉熵（cross entropy）推得了这个固定值的表达式，并且将其称之为局部拉德马赫正则化项，并加在经验损失函数上。将正则化方法作用在混合和模型集成之后，得到了CIFAR-10上目前最好的准确率。

## 安装

在当前目录下运行样例代码需要PadddlePaddle Fluid的v.1.3.0或以上的版本。如果你的运行环境中的PaddlePaddle低于此版本，请根据[安装文档](http://www.paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html#paddlepaddle)中的说明来更新PaddlePaddle。

## 数据准备

第一次使用CIFAR-10数据集时，您可以通过如果命令下载：

    sh ./dataset/download.sh

请确保您的环境有互联网连接。数据会下载到`train.py`同目录下的`dataset/cifar/cifar-10-batches-py`。如果下载失败，您可以自行从 https://www.cs.toronto.edu/~kriz/cifar.html 上下载cifar-10-python.tar.gz并解压到上述位置。

## 模型训练

数据准备好后，可以通过如下命令开始训练：

    sh run_cifar.sh

- 在```run_cifar.sh```中通过设置 ```export CUDA_VISIBLE_DEVICES=0```指定GPU卡号进行训练。
- 可选参数见：

    python train_mixup.py --help

**数据读取器说明：**

* 数据读取器定义在`reader_cifar.py`中
* 输入图像尺寸统一变换为32 * 32
* 训练时将图像填充为40 * 40然后随机剪裁为原输入图像大小
* 训练时图像随机水平翻转
* 对图像每个像素做归一化处理
* 训练时对图像做随机遮挡
* 训练时对输入图像做随机洗牌

**模型配置：**

* 采用momentum优化算法训练，momentum=0.9
* 总共训练600轮
* 对梯度采用全局L2范数裁剪
* 其余模型配置在run_cifar.sh中

## 模型测试

可以通过如下命令开始测试：

    sh run_cifar_test.sh

- 在```run_cifar_test.sh```中通过设置 ```export CUDA_VISIBLE_DEVICES=0```指定GPU卡号进行训练。
- 可选参数见：

    python test_mixup.py --help

得到六个模型后运行如下脚本得到融合模型：

    python voting.py


## 实验结果

下表为模型评估结果：

| 模型                   |   初始学习率  | 批量大小   | 模型编号   | acc-1  |
| :--------------- | :--------: | :------------:    | :------------------:    |------: |
| [model_0](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_0.tar.gz)  | 0.01 | 64  | 0 | 97.12% |
| [model_1](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_1.tar.gz)  | 0.02 | 80  | 0 | 97.34% |
| [model_2](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_2.tar.gz)  | 0.015 | 80 | 1 | 97.31% |
| [model_3](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_3.tar.gz)  | 0.02 | 80  | 1 | 97.52% |
| [model_4](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_4.tar.gz)  | 0.03 | 80  | 1 | 97.30% |
| [model_5](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_5.tar.gz)  | 0.015 | 64 | 2 | 97.32% |

融合模型acc-1=98.01%

## 引用

  - DARTS: Differentiable Architecture Search [`论文`](https://arxiv.org/abs/1806.09055)
  - Differentiable Architecture Search in PyTorch [`代码`](https://github.com/quark0/darts)
