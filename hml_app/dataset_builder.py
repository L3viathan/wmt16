import numpy as np
import os
import sys

import autopath
from wmt16.ml_app.utils.app_funs import get_domain, read_lett

#TODO: set these constants
debug=True
vector_size = 6 #TODO: set it, the size of feature for a candidate pair
train_dir_path = '../ml_app/data/lett.train'

def build_dataset(corpus_file):
    '''Build full size matrix for a corpus e.g. train/valid/test.
    Eache row is feature for a pair of documents.
    '''

    print '----------Build dataset:', corpus_file
    lines = []
    with open(corpus_file, 'rt') as f:
        lines = f.readlines()

    num = len(lines)
    X = np.zeros(shape=(num, vector_size))
    Y = np.zeros(shape=(num,))

    old_domain = ''
    for idx, line in enumerate(lines):
        en_url, fr_url, label = line.split('\t')

        domain = get_domain(en_url)
        if old_domain != domain:
            print '---build feature for:', domain
            en_corpus, fr_corpus = read_lett(os.path.join(train_dir_path, domain + '.lett.gz'), 'en', 'fr')
            old_domain = domain

        en_page = en_corpus[en_url]
        fr_page = fr_corpus[fr_url]

        Y[idx] = int(label)
        features = [0, 1, 2, 3, 4, 5]#TODO: your feature for a candidate pair, en_url, fr_url 
        X[idx] = features

        if debug and idx>=100: break

    return X, Y

def get_one_url_corpus(en_url):
    '''Get features for the given en_url page and all fr_url candidate.'''

    domain = get_domain(en_url)
    en_corpus, fr_corpus = read_lett(os.path.join(train_dir_path, domain + '.lett.gz'), 'en', 'fr')

    all_candidates = fr_corpus.keys()
    #TODO: list of most potential fr_url candidates for the given en_url
    fr_candidates = all_candidates[0:2]


    num = len(fr_candidates)
    X = np.zeros(shape=(num, vector_size))
    fr_urls = []
    for idx, fr_url in enumerate(fr_candidates):

        en_page = en_corpus[en_url]
        fr_page = fr_corpus[fr_url]
        features = [0, 1, 2, 3, 4, 5]#TODO: your feature for a candidate pair, en_url, fr_url 

        X[idx] = features
        fr_urls.append(fr_url)
        
    return X, fr_urls

def main():
    corpus_file = '../ml_app/data/valid_enriched10.txt'
    X, Y = build_dataset(corpus_file)

    en_url = 'http://bugadacargnel.com/en/pages/artistes.php?name=annikalarsson'
    Xone, fr_urls = get_one_url_corpus(en_url)

    print X.shape, Y.shape, Xone.shape, len(fr_urls)

if __name__ == '__main__': 
    main()

