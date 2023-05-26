import math

import numpy as np
import pandas as pd
from torchtext.data import BucketIterator
from underthesea import word_tokenize
from main import set_SEED, parse_args, Checkpoint
from parser_data.load_data import load_json
from parser_data.prepare_data import HandleDataset
from seq2seq.metrics import BleuScorer
from seq2seq.models import PAD_TOKEN
from seq2seq.models.seq2seq import Seq2Seq
import nltk

from seq2seq.prediction import Predictor
from seq2seq.trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim

nltk.download('wordnet')

set_SEED()

train = load_json('datasets/ViNewsQA/train_ViNewsQA.json')
val = load_json('datasets/ViNewsQA/dev_ViNewsQA.json')
test = load_json('datasets/ViNewsQA/test_ViNewsQA.json')

dataset = HandleDataset(train, val, test)
dataset.load_data_and_fields()
src_vocab, trg_vocab = dataset.get_vocabs()
train_data, valid_data, test_data = dataset.get_data()

print('--------------------------------')
print(f"Training data: {len(train_data.examples)}")
print(f"Evaluation data: {len(valid_data.examples)}")
print(f"Testing data: {len(test_data.examples)}")
print('--------------------------------')
print(f'Input example: {train_data.examples[4].src}\n')
print(f'Output example: {train_data.examples[4].trg}')
print('--------------------------------')
print(f"Unique tokens in questions vocabulary: {len(src_vocab)}")
print(f"Unique tokens in answers vocabulary: {len(trg_vocab)}")
print('--------------------------------')

args = parse_args()

RNN_NAME = 'rnn'
CNN_NAME = 'cnn'
TRANSFORMER_NAME = 'transformer'

ATTENTION_1 = 'bahdanau'
ATTENTION_2 = 'luong'
# Choose model here
args.model = CNN_NAME # CNN and Transformers don't apply Attention_1, Attention_2
args.attention = ATTENTION_1
cell_name = 'gru'

if args.model == RNN_NAME and args.attention == ATTENTION_1:
    from seq2seq.models.rnn1 import Encoder, Decoder
elif args.model == RNN_NAME and args.attention == ATTENTION_2:
    from seq2seq.models.rnn2 import Encoder, Decoder
elif args.model == CNN_NAME:
    from seq2seq.models.cnn import Encoder, Decoder
elif args.model == TRANSFORMER_NAME:
    from seq2seq.models.transformer import Encoder, Decoder, NoamOpt

CUDA = 'cuda'
GPU = 'gpu'
CPU = 'cpu'

DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

if args.model == RNN_NAME and args.attention == ATTENTION_2:
    encoder = Encoder(src_vocab, DEVICE, cell_name)
    decoder = Decoder(trg_vocab, DEVICE, cell_name)
else:
    encoder = Encoder(src_vocab, DEVICE)
    decoder = Decoder(trg_vocab, DEVICE)
model = Seq2Seq(encoder, decoder, args.model).to(DEVICE)

parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('--------------------------------')
print(f'Model: {args.model}')
print(f'Model input: {args.input}')
if args.model == RNN_NAME:
    print(f'Attention: {args.attention}')
    print('Cell name: ',cell_name)
print(f'The model has {parameters_num:,} trainable parameters')
print('--------------------------------')

# create optimizer
if args.model ==TRANSFORMER_NAME:
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
else:
    optimizer = optim.Adam(model.parameters(),lr=0.001)

batch_size = 8
epochs=10

# define criterion
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])

trainer = Trainer(optimizer, criterion, batch_size, DEVICE)
train_loss, val_loss = trainer.train(model, train_data, valid_data, 'datasets', num_of_epochs=epochs)

val_ref = [list(filter(None, np.delete([sample["contexts"], sample["answers"], sample["questions"]],[0,1]))) for sample in val]


test_ref = [list(filter(None, np.delete([sample["contexts"], sample["answers"], sample["questions"]],[0,1]))) for sample in test]

val_trg = []
test_trg = []
trg_ = [val_trg,test_trg]
for t in trg_:
    for i in val_ref:
        tmp=[]
        for j in i:
            s = word_tokenize(str(j))
            tmp.append(s)
        t.append(tmp)

val_src = [i.src for i in valid_data.examples]
new_valid = [[val_src[i], val[i]["questions"]] for i in range(len(val))]
test_src = [i.src for i in test_data.examples]
new_test = [[test_src[i], test[i]["questions"]] for i in range(len(test))]



name = args.model+"_"+cell_name if args.model==RNN_NAME else args.model
#model = Checkpoint.load(model,path,'./{}.pt'.format(name))

valid_iterator, test_iterator = BucketIterator.splits(
                                    (valid_data, test_data),
                                    batch_size=8,
                                    sort_within_batch=True if args.model == RNN_NAME else False,
                                    sort_key=lambda x: len(x.src),
                                    device=DEVICE)

# evaluate model
valid_loss = trainer.evaluator.evaluate(model, valid_iterator)
test_loss = trainer.evaluator.evaluate(model, test_iterator)

# calculate blue score for valid and test data
predictor = Predictor(model, src_vocab, trg_vocab, DEVICE)

# # train_scorer = BleuScorer()
valid_scorer = BleuScorer()
test_scorer = BleuScorer()

valid_scorer.data_score(new_valid, predictor)
test_scorer.data_score(new_test, predictor)

print(f'| Val. Loss: {valid_loss:.3f} | Test PPL: {math.exp(valid_loss):7.3f} |')
print(f'| Val. Data Average BLEU1, BLEU4 score {valid_scorer.average_score()} |')
print(f'| Val. Data Average METEOR score {valid_scorer.average_meteor_score()} |')
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
print(f'| Test Data Average BLEU1, BLEU4 score {test_scorer.average_score()} |')
print(f'| Test Data Average METEOR score {test_scorer.average_meteor_score()} |')

r = {'ppl':[round(math.exp(test_loss),3)],
     'BLEU-1':[test_scorer.average_score()[0]*100],
     'BLEU-4':[test_scorer.average_score()[1]*100],
     'METEOR':[test_scorer.average_meteor_score()*100],
     'ROUGE-L':[test_scorer.average_rouge_score()*100]}

df_result = pd.DataFrame(data=r)

print(df_result)

html = df_result.style.set_table_styles([{'selector': 'th', 'props': [('font-size', '15pt')]}]).set_properties(**{'font-size': '15pt'})
