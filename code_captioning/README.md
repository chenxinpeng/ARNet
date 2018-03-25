# Code Captioning

## Prerequisites
 - Python 3.6
 - Pytorch 0.12
 - CUDA 8.0

## Data Pre-processing
The dataset of HabeasCorpus should be existed under the folder `data`, which we also download from [ReviewNet](https://github.com/kimiyoung/review_net/blob/master/code_caption/README.md). Then we pre-process the dataset as follows:
```bash
python3.6 prepro_gen_data.py
```

We can obtain the following data:
 - index2token.pkl
 - token2index.pkl
 - train_data.pkl
 - val_data.pkl
 - test_data.pkl