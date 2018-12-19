# Permuted Sequential MNIST

In this code, we do the experiments based on Tensorflow.

## Update
### Dec. 2018
The reason is that the training speed on GPU is slower than that on CPU with PyTorch 0.1.12. I have no time to find the reasons. So I use TensorFlow for the experiment of sequential MNIST.

## Dependencies
 - Python 2.7
 - Tensorflow >= 1.0
 - Keras 2.0
 - CUDA 8.0

## Training with MLE
```bash
./bash_tf_permuted_mnist_lstm.sh
```

## Fine-tuning with ARNet
```bash
./bash_tf_permuted_mnist_lstm_rcst.sh
```


