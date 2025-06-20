# pt-dec
[![Build Status](https://travis-ci.org/vlukiyanov/pt-dec.svg?branch=master)](https://travis-ci.org/vlukiyanov/pt-dec) [![codecov](https://codecov.io/gh/vlukiyanov/pt-dec/branch/master/graph/badge.svg)](https://codecov.io/gh/vlukiyanov/pt-dec)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5877a6b3baa342c3bb2d8f4a4c94f8dd)](https://app.codacy.com/app/vlukiyanov/pt-dec?utm_source=github.com&utm_medium=referral&utm_content=vlukiyanov/pt-dec&utm_campaign=Badge_Grade_Settings)

This repository provides a PyTorch implementation of a variant of the Deep Embedded Clustering (DEC) algorithm. It is compatible with PyTorch 1.0.0 and supports Python versions 3.6 and 3.7, with or without CUDA acceleration.

To set up the environment for running this code, you can create a new Conda environment with Python 3.7 using the following command:

```bash
conda create --name your_env_name python=3.7
```
Replace `your_env_name` with a name of your choice for the environment.

Once your env activated, run:
```bash
pip install -r requirements.txt
```

This follows (*or attempts to; note this implementation is unofficial*) the algorithm described in "Unsupervised Deep Embedding for Clustering Analysis" of Junyuan Xie, Ross Girshick, Ali Farhadi (<https://arxiv.org/abs/1511.06335>).

## Examples

An extra example using MNIST data can be found in the `examples/mnist/mnist.py` which achieves around 85% accuracy.

## Usage

This is distributed as a Python package `ptdec` and can be installed with `python setup.py install` after installing `ptsdae` from https://github.com/vlukiyanov/pt-sdae. The PyTorch `nn.Module` class representing the DEC is `DEC` in `ptdec.dec`, while the `train` function from `ptdec.model` is used to train DEC.

## Other implementations of DEC

*   Original Caffe: <https://github.com/piiswrong/dec>
*   PyTorch: <https://github.com/CharlesNord/DEC-pytorch> and <https://github.com/eelxpeng/dec-pytorch>
*   Keras: <https://github.com/XifengGuo/DEC-keras> and <https://github.com/fferroni/DEC-Keras>
*   MXNet: <https://github.com/apache/incubator-mxnet/blob/master/example/deep-embedded-clustering/dec.py>
*   Chainer: <https://github.com/ymym3412/DeepEmbeddedClustering>
