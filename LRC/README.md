# LRC Local Rademachar Complexity Regularization
Regularization of Deep Neural Networks(DNNs) for the sake of improving their generalization capability is important and chllenging. This directory contains image classification model based on a novel regularizer rooted in Local Rademacher Complexity (LRC). We appreciate the contribution by [DARTS](https://arxiv.org/abs/1806.09055) for our research. The regularization by LRC and DARTS are combined in this model to reach accuracy of 98.01% on CIFAR-10 dataset. Code accompanying the paper
> [An Empirical Study on Regularization of Deep Neural Networks by Local Rademacher Complexity](https://arxiv.org/abs/1902.00873)\
> Yingzhen Yang, Xingjian Li, Jun Huan.\
> _arXiv:1902.00873_.

---
# Table of Contents

- [Introduction of algorithm](#introduction-of-algorithm)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Testing](#testing)
- [Experimental result](#experimental-result)
- [Reference](#reference)

## Introduction of algorithm

Rademacher complexity is well known as a distribution-free complexity measure of function class and LRC focus on a restricted function class which leads to sharper convergence rates and potential better generalization. Our LRC based regularizer is developed by estimating the complexity of the function class centered at the minimizer of the empirical loss of DNNs.

## Installation

Running sample code in this directory requires PaddelPaddle Fluid v.1.3.0 and later. If the PaddlePaddle on your device is lower than this version, please follow the instructions in [installation document](http://www.paddlepaddle.org/documentation/docs/zh/1.3/beginners_guide/install/index_cn.html#paddlepaddle) and make an update.

## Data preparation

When you want to use the cifar-10 dataset for the first time, you can download the dataset as:

    sh ./dataset/download.sh

Please make sure your environment has an internet connection.

The dataset will be downloaded to `dataset/cifar/cifar-10-batches-py` in the same directory as the `train.py`. If automatic download fails, you can download cifar-10-python.tar.gz from https://www.cs.toronto.edu/~kriz/cifar.html and decompress it to the location mentioned above.


## Training

After data preparation, one can start the training step by:

    sh run_cifar.sh

- Set ```export CUDA_VISIBLE_DEVICES=0``` to specifiy one GPU to train.
- For more help on arguments:

    python train_mixup.py --help

**data reader introduction:**

* Data reader is defined in `reader_cifar.py`.
* Reshape the images to 32 * 32.
* In training stage, images are padding to 40 * 40 and cropped randomly to the original size.
* In training stage, images are horizontally random flipped.
* Images are standardized to (0, 1).
* In training stage, cutout images randomly.
* Shuffle the order of the input images during training.

**model configuration:**

*  Use momentum optimizer with momentum=0.9.
*  Total epoch is 600.
*  Use global L2 norm to clip gradient.
*  Other configurations are set in `run_cifar.sh`

## Tesing

one can start the testing step by:

    sh run_cifar_test.sh

- Set ```export CUDA_VISIBLE_DEVICES=0``` to specifiy one GPU to train.
- For more help on arguments:

    python test_mixup.py --help

After obtaining six models, one can get ensembled model by:

    python voting.py

## Experimental result

Experimental result is shown as below:

| Model                   |   based lr  | batch size | model id  | acc-1  |
| :--------------- | :--------: | :------------:    | :------------------:    |------: |
| [model_0](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_0.tar.gz)  | 0.01 | 64  | 0 | 97.12% |
| [model_1](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_1.tar.gz)  | 0.02 | 80  | 0 | 97.34% |
| [model_2](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_2.tar.gz)  | 0.015 | 80 | 1 | 97.31% |
| [model_3](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_3.tar.gz)  | 0.02 | 80  | 1 | 97.52% |
| [model_4](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_4.tar.gz)  | 0.03 | 80  | 1 | 97.30% |
| [model_5](https://paddlemodels.bj.bcebos.com/autodl/lrc_model_5.tar.gz)  | 0.015 | 64 | 2 | 97.32% |

ensembled model acc-1=98.01%

## Reference

  - DARTS: Differentiable Architecture Search [`paper`](https://arxiv.org/abs/1806.09055)
  - Differentiable architecture search in PyTorch [`code`](https://github.com/quark0/darts)
