# Question Generation (QG): An Experimental Study for Vietnamese Text

Authors: Huu-Loi Le et al

## Directory
Please note that you should prepare a folder to store the data as shown below, to avoid errors during the model training process.

    ├── datasets/
      ├── ViNewsQA/
        ├── train.json
        ├── dev.jon
        ├── test.jon
      ├── ViQuAD/
    ├── parser_data/
    ├── seq2seq/
    ├── cli.py
    └── main.py

## Data
The available datasets for this source code include: [ViNewsQA](https://arxiv.org/abs/2006.11138), [ViQuAD](https://arxiv.org/abs/2009.14725), 
[ViCoQA](https://arxiv.org/abs/2105.01542), [ViMMRC1.0](https://arxiv.org/abs/2008.08810), and [ViMMRC2.0](https://arxiv.org/abs/2303.18162).

## Usage
### Install
```
git clone https://github.com/Shaun-le/ViQG.git
cd ViQG
```
### Prerequisite
To install dependencies, run:
```
pip install -r requirements.txt
```
### CLI
To proceed with model training, please run the following code snippets:
- RNN-1 [(Bahdanau et al., 2016)](https://arxiv.org/abs/1409.0473)
```
python cli.py evaluate --model_name 'rnn' --dataset [*] --attention 'bahdanau'
```
- RNN-2 [(Luong et al., 2015)](https://arxiv.org/abs/1508.04025)
```
python cli.py evaluate --model_name 'rnn' --dataset [*] --attention 'luong'
```
- Convolutional [(Gehring et al., 2017)](https://arxiv.org/abs/1705.03122)
```
python cli.py evaluate --model_name 'cnn' --dataset [*]
```
- Transformer [(Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
```
python cli.py evaluate --model_name 'transformer' --dataset [*]
```
- ViT5 and BARTpho
```
Comming soon!
```
***Note**

[*]: name of dataset

If you want to customize the batch size and number of epochs, you can do the following:
```
python cli.py evaluate --model_name 'rnn' --dataset [*] --attention 'bahdanau' --batch_size 16 --epochs_num 10
```

## Citation

    Comming soon!
