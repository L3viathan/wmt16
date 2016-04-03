import random

import autopath
from ml_app.utils.io_funs import text_file_line_iter

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def divide(train=0.6, valid=0.2, test=0.2):
    train_file = '../data/train.pairs'
    train_set = []
    valid_set = []
    test_set = []
    for line in text_file_line_iter(train_file):
        rnd = random.random()
        if rnd<train:
            train_set.append(line)
        elif rnd<train+valid:
            valid_set.append(line)
        else:
            test_set.append(line)
    
    with open('../data/train.txt', 'wt') as f:
        f.write(''.join(train_set))
    with open('../data/valid.txt', 'wt') as f:
        f.write(''.join(valid_set))
    with open('../data/test.txt', 'wt') as f:
        f.write(''.join(test_set))

def main():
    get_config()
    divide()

if __name__ == '__main__':
    main()
