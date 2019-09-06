# 基于神经网络搜索的图片分类

该项目包含了10种图片分类模型。其中的九种模型是由不同的神经网络搜索(NAS)算法自动搜索出来的，另外一种是ResNet模型。
我们提供了代码和训练脚本来在CIFAR-10和CIFAR-100数据集上训练和测试这些模型。
在训练过程中，我们使用了标准的数据增强技术，即随机裁剪，随机翻转，和归一化。

---
## 内容概括
- [安装说明](#安装说明)
- [数据准备](#数据准备)
- [训练模型](#训练模型)
- [项目文件结构介绍](#项目文件结构)
- [引用](#引用)


### 安装说明
这个项目依赖于以下一些软件包：
- Python = 3.6
- PadddlePaddle Fluid >= v0.15.0
- numpy, tarfile, cPickle, PIL


### 数据准备
请在运行代码前下载 [CIFAR-10](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz) 和 [CIFAR-100](https://dataset.bj.bcebos.com/cifar/cifar-100-python.tar.gz)。
请注意CIFAR-10-Python压缩文件的MD5值是`c58f30108f718f92721af3b95e74349a`，CIFAR-100-Python压缩文件的MD5值是`eb9058c3a382ffc7106e4002c42a8d85`。
请将这两个下载文件保存在`${TORCH_HOME}/cifar.python`路径下。在数据准备之后，应该有两个文件：`${TORCH_HOME}/cifar.python/cifar-10-python.tar.gz`和`${TORCH_HOME}/cifar.python/cifar-100-python.tar.gz`。


### Training Models

在设置好环境和准备好数据之后，您可以开始训练模型了。训练的主要入口文件是在`train_cifar.py`中，我们提供了方便的脚本可以直接训练，如下：
```
bash ./scripts/base-train.sh 0 cifar-10 ResNet110
bash ./scripts/train-nas.sh  0 cifar-10 GDAS_V1
bash ./scripts/train-nas.sh  0 cifar-10 GDAS_V2
bash ./scripts/train-nas.sh  0 cifar-10  SETN
bash ./scripts/train-nas.sh  0 cifar-10 NASNet
bash ./scripts/train-nas.sh  0 cifar-10 ENASNet
bash ./scripts/train-nas.sh  0 cifar-10 AmoebaNet
bash ./scripts/train-nas.sh  0 cifar-10 PNASNet
bash ./scripts/train-nas.sh  0 cifar-100 SETN
```
第一个参数指定在哪块GPU上运行该的程序(GPU-ID)，第二个参数指定数据集名称(`cifar-10`或`cifar-100`)，第三个参数是指定了模型名称。
如果您要训练ResNet模型，请使用`./scripts/base-train.sh`；如果您要训练NAS搜索出的模型，请使用`./scripts/train-nas.sh`。


### 项目文件结构
```
.
├──train_cifar.py [训练卷积神经网络模型的文件]
├──lib [数据集，模型，及其他相关库]
│  └──models  
│     ├──__init__.py [引用一些模型相关的函数和类]
│     ├──resnet.py [定义ResNet模型]
│     ├──operations.py [定义了NAS搜索空间中的一些原子级操作]
│     ├──genotypes.py [定义了不同的NAS搜索出的模型的拓扑结构]
│     └──nas_net.py [定义了NAS模型的宏观结构]
│  └──utils
│     ├──__init__.py [引用一些辅助模块]
│     ├──meter.py [定义了AverageMeter类来统计模型的准确率和损失函数值]
│     ├──time_utils.py [定义了打印时间和转换时间度量的函数]
│     └──data_utils.py [定义了数据集相关的读取和数据增强相关的函数]
└──scripts [运行脚本]
```


### 引用
如果您发现这个项目对您的研究有帮助，请考虑引用下面的某些论文：
```
@inproceedings{dong2019one,
  title     = {One-Shot Neural Architecture Search via Self-Evaluated Template Network},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year      = {2019}
}
@inproceedings{dong2019search,
  title     = {Searching for A Robust Neural Architecture in Four GPU Hours},
  author    = {Dong, Xuanyi and Yang, Yi},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1761--1770},
  year      = {2019}
}
@inproceedings{liu2018darts,
  title     = {Darts: Differentiable architecture search},
  author    = {Liu, Hanxiao and Simonyan, Karen and Yang, Yiming},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2018}
}
@inproceedings{pham2018efficient,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and Guan, Melody and Zoph, Barret and Le, Quoc and Dean, Jeff},
  booktitle = {International Conference on Machine Learning (ICML)},
  pages     = {4092--4101},
  year      = {2018}
}
@inproceedings{liu2018progressive,
  title     = {Progressive neural architecture search},
  author    = {Liu, Chenxi and Zoph, Barret and Neumann, Maxim and Shlens, Jonathon and Hua, Wei and Li, Li-Jia and Fei-Fei, Li and Yuille, Alan and Huang, Jonathan and Murphy, Kevin},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  pages     = {19--34},
  year      = {2018}
}
@inproceedings{zoph2018learning,
  title     = {Learning transferable architectures for scalable image recognition},
  author    = {Zoph, Barret and Vasudevan, Vijay and Shlens, Jonathon and Le, Quoc V},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {8697--8710},
  year      = {2018}
}
@inproceedings{real2019regularized,
  title     = {Regularized evolution for image classifier architecture search},
  author    = {Real, Esteban and Aggarwal, Alok and Huang, Yanping and Le, Quoc V},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  pages     = {4780--4789},
  year      = {2019}
}
```
