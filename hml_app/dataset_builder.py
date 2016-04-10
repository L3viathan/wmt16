import numpy as np
import os
import sys

import autopath
from wmt16.ml_app.utils.app_funs import get_domain, read_lett
from wmt16 import lexical_filter

#TODO: set these constants
debug=False
vector_size = 4 #TODO: set it, the size of feature for a candidate pair
#train_dir_path = '/tmp/u/vutrongh/lett.train'
train_dir_path = '../ml_app/data/lett.train'
LEN_UPPER_BOUND = 1.3
LEN_LOWER_BOUND = 0.7
COMMON_WORDS_BOUND = 0.02

import pdb

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
        print fr_url

        domain = get_domain(en_url)
        if old_domain != domain:
            en_corpus, fr_corpus = lexical_filter.read_lett(os.path.join(train_dir_path, domain + '.lett.gz'), 'en', 'fr')
            print '---build feature for:', domain
            old_domain = domain

        en_page = en_corpus[en_url]
        fr_page = fr_corpus[fr_url]

        Y[idx] = int(label)
        #features = [0, 1, 2, 3, 4, 5]#TODO: your feature for a candidate pair, en_url, fr_url 
        features = get_vector(en_page, fr_page)
        print features
        if features != None:
            pdb.set_trace()
        X[idx] = features

        if debug and idx>=10: break

    return X, Y

def get_vector(en_page, fr_page):
    len_rate = en_page.length/fr_page.length
    if(LEN_UPPER_BOUND < len_rate or len_rate < LEN_LOWER_BOUND): return None
    inter = set(en_page.constants).intersection(fr_page.constants) # intersection without indexes
    inter_rate = len(inter)/float(en_page.length)
    if(inter_rate < 0.015): return None
    # not cache occur map yet for now.
    occur_map1, map1 = None, None
    inter, occur_map1, map1 = lexical_filter.doc_inter(en_page.tokens, fr_page.tokens, occur_map1, map1)
    inter_sim = lexical_filter.doc_similarity_pos_aware(inter, en_page.length, fr_page.length)
    if(inter_sim < 0.50): return None
    sent_rate = en_page.sent_length /float(fr_page.sent_length)
    return [len_rate, inter_rate, inter_sim, sent_rate]#length rate, shared word/en _lang, distance/max(doc_len), sentence rate

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
        features = [0, 1, 2, 3]#TODO: your feature for a candidate pair, en_url, fr_url 

        X[idx] = features
        fr_urls.append(fr_url)
        
    return X, fr_urls

def main():
    corpus_file = '../ml_app/thanh_data/valid_enriched10.txt'
    X, Y = build_dataset(corpus_file)
    pdb.set_trace()

    en_url = 'http://bugadacargnel.com/en/pages/artistes.php?name=annikalarsson'
    Xone, fr_urls = get_one_url_corpus(en_url)

    print X.shape, Y.shape, Xone.shape, len(fr_urls)

if __name__ == '__main__': 
    main()

