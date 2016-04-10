import random

import autopath
from ml_app.utils.io_funs import text_file_line_iter

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def divide(train=0.6, valid=0.2, test=0.2):
    '''Divide random data from train.pair.'''

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

def get_url_from_line(line):
    return line.split('\t')[0]

def get_source_url(train_file):
    urls = []
    for line in text_file_line_iter(train_file):
        line = line.strip()
        url = get_url_from_line(line)
        if url not in urls:
            urls.append(url)
    return urls

def divide_filtered_data():
    train_file = '../shared_data/original_data/train.txt'
    valid_file = '../shared_data/original_data/valid.txt'
    test_file = '../shared_data/original_data/test.txt'
    filtered_file = '../shared_data/top10_pairs/all_top10_candidate_pairs.txt'

    train_en_urls = get_source_url(train_file)
    valid_en_urls = get_source_url(valid_file)
    test_en_urls = get_source_url(test_file)

    train_set = []
    valid_set = []
    test_set = []
    for line in text_file_line_iter(filtered_file):
        line = line.strip()
        url = get_url_from_line(line)
        if url in train_en_urls:
            train_set.append(line)
        elif url in valid_en_urls:
            valid_set.append(line)
        elif url in test_en_urls:
            test_set.append(line)
        else:
            print '!!!!! not in anyset??', url
            test_en_urls.append(line)
    
    with open('./train_top10.txt', 'wt') as f:
        f.write('\n'.join(train_set))
    with open('./valid_top10.txt', 'wt') as f:
        f.write('\n'.join(valid_set))
    with open('./test_top10.txt', 'wt') as f:
        f.write('\n'.join(test_set))

def main():
    get_config()
    #divide()
    divide_filtered_data()

if __name__ == '__main__':
    main()
