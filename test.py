from main import set_SEED, parse_args
from parser_data.load_data import load_json
from parser_data.prepare_data import HandleDataset
from seq2seq.models import PAD_TOKEN
from seq2seq.models.seq2seq import Seq2Seq
import nltk
from seq2seq.trainer import Trainer
nltk.download('wordnet')
import torch
import torch.nn as nn
import torch.optim as optim

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
epochs=1

# define criterion
criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])

trainer = Trainer(optimizer, criterion, batch_size, DEVICE)
train_loss, val_loss = trainer.train(model, train_data, valid_data, 'datasets', num_of_epochs=epochs)