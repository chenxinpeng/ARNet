# Regularizing RNNs for Caption Generation by Reconstructing The Past with The Present (ARNet)

## Introduction
This [paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Chen_Regularizing_RNNs_for_CVPR_2018_paper.html) was accepted by CVPR 2018 as poster. The proposed method is very effective in RNN-based models.  In our framework, RNNs are regularized by reconstructing the previous hidden state with the current one. Therefore, the relationships between neighbouring hidden states in RNNs can be further exploited by our ARNet.

![ARNet framework](https://ws1.sinaimg.cn/large/0069RVTdgy1fu8dkzjjcmj31kw0krdqu.jpg|width=100)

## Experiments
We validate our ARNet on the following tasks:
 - [Image Captioning](http://git.code.oa.com/laviechen/ARNet/tree/master/image_captioning)
 - [Code Captioning](http://git.code.oa.com/laviechen/ARNet/tree/master/code_captioning)
 - [Permuted Sequential MNIST](http://git.code.oa.com/laviechen/ARNet/tree/master/permuted_sequential_mnist)


## License
The code and the models in this repo are released under the CC-BY-NC 4.0 LICENSE (refer to the LICENSE file for details).


## Citation

    @InProceedings{Chen_2018_CVPR,
      author = {Chen, Xinpeng and Ma, Lin and Jiang, Wenhao and Yao, Jian and Liu, Wei},
      title = {Regularizing RNNs for Caption Generation by Reconstructing the Past With the Present},
      booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2018}
    }


## Authorship
This project is maintained by [Xinpeng Chen](https://chenxinpeng.github.io/).