# pt-dec
[![Build Status](https://travis-ci.org/vlukiyanov/pt-dec.svg?branch=master)](https://travis-ci.org/vlukiyanov/pt-dec) [![codecov](https://codecov.io/gh/vlukiyanov/pt-dec/branch/master/graph/badge.svg)](https://codecov.io/gh/vlukiyanov/pt-dec)


This repository offers a PyTorch implementation of a variant of the Deep Embedded Clustering (DEC) algorithm. The original code can be found at [vlukiyanov/pt-dec](https://github.com/vlukiyanov/pt-dec/tree/master/ptdec). This implementation is compatible with PyTorch 1.0.0 and supports Python 3.6 and 3.7, with optional CUDA acceleration.

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

To run `tcc.py` with customizable options, use the following command-line flags:

```bash
python tcc.py [OPTIONS]
```

Available options:

- `--cuda`: Use CUDA for acceleration (default: `False`)
- `--testing-mode`: Run in testing mode (default: `False`)
- `--train-autoencoder`: Whether to train the autoencoder from scratch or load an existing one (default: `True`)
- `--sort-by-elem`: Whether to split the data by its "ElemDespesaTCE" and cluster each part separately (default: `False`)

**Example usage:**

```bash
python tcc.py --train-autoencoder False --sort-by-elem True
```



## Other implementations of DEC

*   Original Caffe: <https://github.com/piiswrong/dec>
*   PyTorch: <https://github.com/CharlesNord/DEC-pytorch> and <https://github.com/eelxpeng/dec-pytorch>
*   Keras: <https://github.com/XifengGuo/DEC-keras> and <https://github.com/fferroni/DEC-Keras>
*   MXNet: <https://github.com/apache/incubator-mxnet/blob/master/example/deep-embedded-clustering/dec.py>
*   Chainer: <https://github.com/ymym3412/DeepEmbeddedClustering>
