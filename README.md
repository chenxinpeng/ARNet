# Regularizing RNNs for Caption Generation by Reconstructing The Past with The Present (ARNet)

## Introduction
This [paper](https://chenxinpeng.github.io/publication/ARNet.pdf) was accepted by CVPR 2018. The proposed method is very effective in RNN-based models.  In our framework, RNNs are regularized by reconstructing the previous hidden state with the current one. Therefore, the relationships between neighbouring hidden states in RNNs can be further exploited by our ARNet.


We validate our ARNet on the following tasks::
 - [Image Captioning](https://github.com/chenxinpeng/ARNet/tree/master/image_captioning)
 - [Code Captioning](https://github.com/chenxinpeng/ARNet/tree/master/code_captioning)
 - [Permuted Sequential MNIST](https://github.com/chenxinpeng/ARNet/tree/master/permuted_sequential_mnist)


## Citation

    @article{chen2018arnet,
      title={Regularizing RNNs for Caption Generation by Reconstructing The Past with The Present},
      author={Chen, Xinpeng and Ma, Lin and Jiang, Wenhao and Yao, Jian and Liu, Wei},
      booktitle={CVPR},
      year={2018}
    }

## License
ARNet is released under the MIT License (refer to the LICENSE file for details).


## Authorship
This project is maintained by [Xinpeng Chen](https://chenxinpeng.github.io/).