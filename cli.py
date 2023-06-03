import click
import math
import numpy as np
import pandas as pd
from torchtext.data import BucketIterator
from underthesea import word_tokenize
from main import set_SEED
from parser_data.load_data import load_json
from parser_data.prepare_data import HandleDataset
from seq2seq.metrics import ComputeScorer
from seq2seq.models.conf import PAD_TOKEN
from seq2seq.models.seq2seq import Seq2Seq
from seq2seq.prediction import Predictor
from seq2seq.trainer import Trainer
import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import display
import nltk
nltk.download('wordnet')


@click.group()
def cli():
    pass

@cli.command('evaluate')
@click.option('--model_name', type=click.Choice(('rnn','cnn','transformer')), default=None,
              help="Choice model")
@click.option('--dataset', type=click.Choice(('ViNewsQA','ViQuAD','ViCoQA','ViMMRC1.0','ViMMRC2.0')),
                default=None, help="the dataset used for training model")
@click.option('--attention', default='luong', type=click.Choice(('bahdanau','luong')), help='attention layer for rnn model')
@click.option('--batch_size', default=8, type=int, help='batch size')
@click.option('--epochs_num', default=20, type=int, help='number of epochs')
@click.option('--cell_name', type=click.Choice(('lstm','gru')), default='gru')
def _evaluate(model_name, dataset, attention, batch_size, epochs_num, cell_name):
    """
    Training and evaluate model for QG task in Vietnamese Text
    """

    print("data: ", dataset)
    print("model: ", model_name)
    print('--------------------------------')
    train = load_json(f'datasets/{dataset}/train.json', dataset)
    val = load_json(f'datasets/{dataset}/dev.json', dataset)
    test = load_json(f'datasets/{dataset}/test.json', dataset)
    dataset = HandleDataset(train, val, test)
    dataset.load_data_and_fields()
    src_vocab, trg_vocab = dataset.get_vocabs()
    train_data, valid_data, test_data = dataset.get_data()
    print('--------------------------------')
    print(f"Training data: {len(train_data.examples)}")
    print(f"Evaluation data: {len(valid_data.examples)}")
    print(f"Testing data: {len(test_data.examples)}")
    print('--------------------------------')
    print(f"Unique tokens in questions vocabulary: {len(src_vocab)}")
    print(f"Unique tokens in answers vocabulary: {len(trg_vocab)}")
    print('--------------------------------')

    set_SEED()

    if model_name == 'rnn' and attention == 'bahdanau':
        from seq2seq.models.rnn1 import Encoder, Decoder
    elif model_name == 'rnn' and attention == 'luong':
        from seq2seq.models.rnn2 import Encoder, Decoder
    elif model_name == 'cnn':
        from seq2seq.models.cnn import Encoder, Decoder
    elif model_name == 'transformer':
        from seq2seq.models.transformer import Encoder, Decoder, NoamOpt

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'rnn' and attention == 'luong':
        encoder = Encoder(src_vocab, DEVICE, cell_name)
        decoder = Decoder(trg_vocab, DEVICE, cell_name)
    else:
        encoder = Encoder(src_vocab, DEVICE)
        decoder = Decoder(trg_vocab, DEVICE)

    model = Seq2Seq(encoder, decoder, model_name).to(DEVICE)

    parameters_num = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('--------------------------------')
    print(f'Model: {model_name}')
    print(f'Model input: context+answer')
    if model_name == 'rnn':
        print(f'Attention: {attention}')
        print('Cell name: ', cell_name)
    print(f'The model has {parameters_num:,} trainable parameters')
    print('--------------------------------')
    # create optimizer
    if model_name == 'transformer':
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)


    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab.stoi[PAD_TOKEN])
    trainer = Trainer(optimizer, criterion, batch_size, DEVICE)
    train_loss, val_loss = trainer.train(model, train_data, valid_data, 'datasets', num_of_epochs=epochs_num)

    val_ref = [list(filter(None, np.delete([sample["contexts"], sample["answers"], sample["questions"]], [0, 1]))) for
               sample in val]

    test_ref = [list(filter(None, np.delete([sample["contexts"], sample["answers"], sample["questions"]], [0, 1]))) for
                sample in test]

    val_trg = []
    test_trg = []
    trg_ = [val_trg, test_trg]
    for t in trg_:
        for i in test_ref:
            tmp = []
            for j in i:
                s = word_tokenize(str(j))
                tmp.append(s)
            t.append(tmp)

    val_src = [i.src for i in valid_data.examples]
    new_valid = [[val_src[i], [word_tokenize(val[i]["questions"])]] for i in range(len(val))]
    test_src = [i.src for i in test_data.examples]
    new_test = [[test_src[i], [word_tokenize(test[i]["questions"])]] for i in range(len(test))]

    valid_iterator, test_iterator = BucketIterator.splits(
        (valid_data, test_data),
        batch_size=8,
        sort_within_batch=True if model_name == 'rnn' else False,
        sort_key=lambda x: len(x.src),
        device=DEVICE)

    # evaluate model
    valid_loss = trainer.evaluator.evaluate(model, valid_iterator)
    test_loss = trainer.evaluator.evaluate(model, test_iterator)

    # calculate blue score for valid and test data
    predictor = Predictor(model, src_vocab, trg_vocab, DEVICE)

    valid_scorer = ComputeScorer()
    test_scorer = ComputeScorer()

    valid_scorer.data_score(new_valid, predictor)
    test_scorer.data_score(new_test, predictor)

    print(f'| Val. Loss: {valid_loss:.3f} | Test PPL: {math.exp(valid_loss):7.3f} |')
    print(f'| Val. Data Average BLEU1,BLEU2, BLEU3, BLEU4 score {valid_scorer.average_score()} |')
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print(f'| Test Data Average BLEU1,BLEU2, BLEU3, BLEU4 score {test_scorer.average_score()} |')

    r = {'ppl': [round(math.exp(test_loss), 3)],
         'BLEU-1': [test_scorer.average_score()[0] * 100],
         'BLEU-2': [test_scorer.average_score()[1] * 100],
         'BLEU-3': [test_scorer.average_score()[2] * 100],
         'BLEU-4': [test_scorer.average_score()[3] * 100],
         'ROUGE-1': [test_scorer.average_rouge_score_n()[0]],
         'ROUGE-2': [test_scorer.average_rouge_score_n()[1]],
         'ROUGE-L': [test_scorer.average_rouge_score() * 100]}

    df_result = pd.DataFrame(data=r)
    df_result.to_csv('results.csv')
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df_result)

if __name__ == '__main__':
    cli()