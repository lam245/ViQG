import argparse
import random
import numpy as np
import torch

"""Constants for the baseline models"""
SEED = 42
QUESTION = 'question'

RNN_NAME = 'rnn'
CNN_NAME = 'cnn'
TRANSFORMER_NAME = 'transformer'

ATTENTION_1 = 'bahdanau'
ATTENTION_2 = 'luong'

SRC_NAME = 'src'
TRG_NAME = 'trg'

path = '/kaggle/working/'
def parse_args():
    """Add arguments to parser_data"""
    parser = argparse.ArgumentParser(description='Verbalization dataset baseline models.')
    parser.add_argument('--model', default=RNN_NAME, type=str,
                        choices=[RNN_NAME, CNN_NAME, TRANSFORMER_NAME], help='model to train the dataset')
    parser.add_argument('--input', default=QUESTION, type=str,
                        choices=[QUESTION], help='use question as input')
    parser.add_argument('--attention', default=ATTENTION_2, type=str,
                        choices=[ATTENTION_1, ATTENTION_2], help='attention layer for rnn model')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--epochs_num', default=30, type=int, help='number of epochs')
    parser.add_argument('--answer_num', default=1, type=int,
                        choices=[1,2,3,4], help='number of answer')
    args = parser.parse_args()
    return args

def set_SEED():
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Checkpoint(object):
    """Checkpoint class"""
    @staticmethod
    def save(model,cell, path):
        """Save model using name"""
        name_tmp = model.name+"_"+ cell if model.name==RNN_NAME else model.name
        name = f'{name_tmp}.pt'
        torch.save(model.state_dict(), path+name)

    @staticmethod
    def load(model,path, name):
        """Load model using name"""
        #name = f'{model.name}.pt'
        model.load_state_dict(torch.load(path+name))
        return model