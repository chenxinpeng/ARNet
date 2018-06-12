# Image Captioning

## Dependencies
 - Python 3.6
 - Pytorch [0.1.12](http://download.pytorch.org/whl/cu80/torch-0.1.12.post1-cp36-cp36m-linux_x86_64.whl)
 - CUDA 8.0

## Image Feature Extraction
Download the MSCOCO data from [here](http://cocodataset.org/): train2014, val2014, and test2014, respectively. Uncompress and put them into `data/images/mscoco`. For simplicity, we aggregate all the images into one directory (`data/images/mscoco`) using a shell script.

In our paper, we use [Inception-v4](https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_v4.py) network as image encoder, `cd` to `feat_ext/inception_v4`, download the weights of Inception-v4 from [here](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz), then extract the image features as follows:
```bash
CUDA_VISIBLE_DEVICES=0 python3.6 extract_feats_conv_fc_v4.py
```

## Data Pre-processing
You can download and use the data from [here](https://drive.google.com/open?id=1MxKySRCnXN2Q0bBg5Asi_mjJPpNwhtC5):
```bash
unzip -q data.zip
```

You can also pre-process the data by yourself: 
```python
python3.6 prepro_build_vocab.py --word_count_threshold 5
```

Afterwards, you will get the following files:
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


## Caption Evaluation Tools
Download and the `coco-caption` files which are used for evaluation from [here](https://drive.google.com/file/d/14gE7QT29gyPmOGuRDyKo-g0OJw1vEHRN/view?usp=sharing). I have changed these evaluation code into `Python 3.6`since my environment is Python 3.6. Uncompress the `coco-caption` under the current folder.


## Encoder-Decoder Model

### Training with MLE
```bash
./bash_image_caption_ende_xe.sh
```
The pre-trained model can be downloaded from [here](https://drive.google.com/drive/folders/1-l2XY4B_pZT1nrpOwyhs3Z9-QY8MVD30?usp=sharing), put it on the under `./models` folder.

### Fine-tuning with our ARNet
```bash
./bash_image_caption_ende_rcst_lstm.sh
```
The fine-tuned model with our ARNet can be downloaded from [here](https://drive.google.com/drive/folders/1mVNRe_6JCbGixNLpEauJ7DBaJi82XGVT?usp=sharing), put it on the under `./models` folder.

### Inference with Greedy Search
```bash
./bash_image_caption_ende_infg.sh
```

### Inference with Beam Search
```bash
./bash_image_caption_ende_infb.sh
```

### Visualization with t-SNE

First, get the hidden states of the sentences:
```bash
./bash_image_caption_ende_vis.sh
```

Then, visualize the hidden states, for example:
```bash
python3.6 prepro_tsne_reduction_vis.py --vis_batch_size 80 \
                                       --truncation 0  \
                                       --hidden_path models/encoder_decoder_inception_v4_seed_116/model_epoch-33_hidden_states.pkl \
                                       --hidden_reduction_save_path models/encoder_decoder_inception_v4_seed_116/model_epoch-33_hidden_states_reduction.pkl
```


## Attentive Encoder-Decoder Model

### Training with MLE
```bash
./bash_image_caption_soft_att_xe.sh
```

The pre-trained model can be downloaded from from [here](https://drive.google.com/drive/folders/1Gq4nwy-NvkvEjowH9Av6obs96t2yjfor?usp=sharing), put it on the under `./models` folder.

### Fine-tuning with our ARNet
```bash
./bash_image_caption_soft_att_rcst_lstm.sh
```

The fine-tuned model with our ARNet can be downloaded from [here](https://drive.google.com/drive/folders/1TFDvcPMJ1T2KNUjucE4O8mUv4c8V8eXN?usp=sharing) by yourself, put it on the under `./models` folder.

### Inference with Greedy Search
```bash
./bash_image_caption_soft_att_infg.sh
```

### Inference with Beam Search
```bash
./bash_image_caption_soft_att_infb.sh
```

### Visualization with t-SNE

First, get the hidden states of the sentences:
```bash
./bash_image_caption_soft_att_vis.sh
```

Then, visualize the hidden states, for example:
```bash
python3.6 prepro_tsne_reduction_vis.py --vis_batch_size 80 \
                                       --truncation 0  \
                                       --hidden_path models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states.pkl \
                                       --hidden_reduction_save_path models/soft_attention_inception_v4_seed_117/model_epoch-8_hidden_states_reduction.pkl
```