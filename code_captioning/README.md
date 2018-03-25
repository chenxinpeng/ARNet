# Code Captioning

## Dependencies
 - Python 3.6
 - Pytorch 0.12
 - CUDA 8.0

## Data Pre-processing
The dataset of HabeasCorpus should be existed under the folder `data`, which is download from [ReviewNet](https://github.com/kimiyoung/review_net/blob/master/code_caption/README.md). Then we pre-process the dataset as follows:
```bash
python3.6 prepro_gen_data.py
```

We can obtain the following data:
 - index2token.pkl
 - token2index.pkl
 - train_data.pkl
 - val_data.pkl
 - test_data.pkl

## Encoder-Decoder Model

### Training with MLE
To train the encoder-decoder model with MLE, run
```bash
./bash_code_caption_ende_xe.sh
```

### Fine-tuning with ARNet
Then, we fune-tuning the network with our ARNet,
```bash
./bash_code_caption_ende_rcst.sh
```

### Inference
To test the model with greedy search, run
```bash
./bach_code_caption_ende_inf.sh
```

### Visualization
To visualize the hidden states with t-SNE, run
```bash
./bash_code_caption_ende_vis.sh
```
We will get the hidden states of generated sequence. Then, run
```bash
python3.6 prepro_tsne_reduction_vis.py --vis_batch_size 80 \
                                       --truncation 0  \
                                       --hidden_path '...' \
                                       --hidden_reduction_save_path '...'
```


## ReviewNet Model


## Attentive Encoder-Decoder Model
