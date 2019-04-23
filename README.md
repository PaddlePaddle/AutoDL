# Introduction to AutoDL Design

## Content
- [Installation](#Installation)
- [Introduction](#Introduction)
- [Data Preparation](#Data Preparation)
- [Model Training](#Model Training)

## Installation
Running demo code in the current directory requires PadddlePaddle Fluid v.1.3.0 or above. If your runtime environment does not meet this requirement, please update PaddlePaddle according to the documents.
* Install Python2.7
* Install dependencies [PARL](https://github.com/PaddlePaddle/PARL) framework and [absl-py](https://github.com/abseil/abseil-py/tree/master/absl) library，as follows:
```
	pip install parl
	pip install absl-py
```


## Introduction
[AutoDL](http://www.paddlepaddle.org/paddle/ModelAutoDL) is an efficient automated neural architecture design method. It designs quality customized neural architecture via reinforcement learning. The system consists of two components: an encoder of the neural architecture, and a critic of the model performance. The encoder encodes neural architecture using a recurrent neural network, and the critic evaluates the sampled architecture in terms of accuracy, number of model parameters, etc., which are fed back to the encoder. The encoder updates its parameters accordingly, and samples a new batch of architectures. After several iterations, the encoder is trained to converge and finds a quality architecture. The open-sourced AutoDl Design is one implementation of AutoDL technique. Section 2 presents the usage of AutoDL. Section 3 presents the framework and examples.

## Data Preparation
* Clone [PaddlePaddle/AutoDL](https://github.com/PaddlePaddle/AutoDL.git) to local machine，and enter the path of AutoDL Design. 
* Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) training data, unzip to AutoDL Design/cifar, and generate a dataset of 10 classes and 100 images per class using `dataset_maker.py`
```
tar zxf cifar-10-python.tar.gz
python dataset_maker.py
```

## Model Training
In the training process, AutoDLa Design agent generates tokens and adjacency matrices used for training, and the trainer uses these tokens and matrices to construct and train convolutional neural networks. The validation accuracy after 20 epochs are used as feed back for the agent, and the agent updates its policy accordingly. After several iterations, the agent learns to find a quality deep neural network.
![Picture](./AutoDL%20Design/img/cnn_net.png)
Here we provide the following test on the method.

### Test on the convergence of the number of tokens produced
Due to the long training time of CNN, to test the validity of agent framework, we use the number of "correct" tokens produced as a pseudo reward. The agent will learn to produce more "correct" tokens per step. The total length of tokens is set at 20. 
```
	export FLAGS_fraction_of_gpu_memory_to_use=0.98
	export FLAGS_eager_delete_tensor_gb=0.0
	export FLAGS_fast_eager_deletion_mode=1
	CUDA_VISIBLE_DEVICES=0 python -u simple_main.py
```
Expected results：
In the log, `average rewards` gradually converges to 20:

```
Simple run target is 20
mid=0, average rewards=2.500
...
mid=450, average rewards=17.100
mid=460, average rewards=17.000

```

### Training AutoDL to design CNN
Train AutoDL Design on the small scale dataset prepared in the previous section:
```
	export FLAGS_fraction_of_gpu_memory_to_use=0.98
	export FLAGS_eager_delete_tensor_gb=0.0
	export FLAGS_fast_eager_deletion_mode=1
	CUDA_VISIBLE_DEVICES=0 python -u main.py
```
__Note:__ It requires two GPUs for training, GPU used by the Agent is set by `CUDA_VISIBLE_DEVICES=0`(in `main.py`)；Trainer uses GPU set by `CUDA_VISIBLE_DEVICES=1`(in [autodl.py](https://github.com/PaddlePaddle/AutoDL/blob/master/AutoDL%20Design/autodl.py#L124))

Expected results：
In the log, `average accuracy` gradually increases:

```
step = 0, average accuracy = 0.633
step = 1, average accuracy = 0.688
step = 2, average accuracy = 0.626
step = 3, average accuracy = 0.682
......
step = 842, average accuracy = 0.823
step = 843, average accuracy = 0.825
step = 844, average accuracy = 0.808
......
```
### Results

![Picture](./AutoDL%20Design/img/search_result.png)
The x-axis is the number of steps, and the y-axis is validation accuracy of the sampled models. The average performance of the sampled models improves over time.

