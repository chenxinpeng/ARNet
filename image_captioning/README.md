# Image Captioning

## Prerequisites
 - Python 3.6
 - Pytorch 0.12
 - CUDA 8.0

## Image Feature Extraction
Download the MSCOCO data from [here](http://cocodataset.org/): train2014, val2014, and test2014, respectively. Uncompress and put them into `data/images/mscoco`. For simplicity, we aggregate all the images into one directory (`data/images/mscoco`) using a shell script.

In our paper, we use [Inception-v4](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) network as image encoder, `cd` to `feat_ext/inception_v4`, download the weights of Inception-v4 from [here](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz), then extract the image features as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python3.6 extract_feats_conv_fc_v4.py
```

## Caption Data Pre-processing
You can download and use the data from [here](https://drive.google.com/open?id=1MxKySRCnXN2Q0bBg5Asi_mjJPpNwhtC5):
```bash
unzip -q data.zip
```

You can also pre-process the data by yourself: 
```python
python3.6 prepro_build_vocab.py --word_count_threshold 5
```

After this, you will get the following files:
 - word_to_idx.pkl
 - idx_to_word.pkl
 - bias_init_vector.pkl
 - train_images_captions_index.pkl

Finally, make sure that the `data` folder contains the following files:
 - annotations
 - coco-train-idxs.p
 - captions_index.pkl
 - word_to_idx.pkl
 - idx_to_word.pkl


## Encoder-Decoder

### Training with MLE
```bash
./bash_image_caption_ende_xe.sh
```

### Fine-tuning with our ARNet
```bash
./bash_image_caption_ende_rcst_lstm.sh
```

### Inference with Greedy Search
```bash
./bash_image_caption_ende_infg.sh
```

### Inference with Beam Search
```bash
./bash_image_caption_ende_infb.sh
```


## Attentive Encoder-Decoder

### Training with MLE
```bash
./bash_image_caption_soft_att_xe.sh
```

We provide the weights of the model trained by ourselves, you can download it from [here](https://drive.google.com/drive/folders/1Gq4nwy-NvkvEjowH9Av6obs96t2yjfor?usp=sharing), put it on the under `./models` folder.

### Fine-tuning with our ARNet
```bash
./bash_image_caption_soft_att_rcst_lstm.sh
```

We also provide the pre-trained model with ARNet, you can download and run the model from [here](https://drive.google.com/drive/folders/1TFDvcPMJ1T2KNUjucE4O8mUv4c8V8eXN?usp=sharing) by yourself.

### Inference with Greedy Search
```bash
./bash_image_caption_soft_att_infg.sh
```

### Inference with Beam Search
```bash
./bash_image_caption_soft_att_infb.sh
```