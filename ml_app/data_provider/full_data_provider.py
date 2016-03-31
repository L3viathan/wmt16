import pdb

import os
import numpy as np
import time

import autopath
from ml_app.utils.app_funs import get_domain, read_lett

class DataSet(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def next_batch(self, batch_size):
        pass

class FullDataProvider(object):
    '''Everything in memory.'''

    def __init__(self, config):
        self.full_config = config
        self.config = config['data_provider']['FullDataProvider']
        self.fbuilder = self.config['feature_builder'](self.full_config)
    
        self.data_path = self.full_config['general']['data_path']
        self.train_file = self.config['train_file']
        self.valid_file = self.config['valid_file']
        self.test_file = self.config['test_file']

        self.train = self.build_dataset(self.train_file)
        self.valid = None
        self.test = None

    def _read_dataset():#No, prefer to cache only feature of docudment in feature builder
        pass

    def build_dataset(self, corpus_file):
        print '----------Build dataset:', corpus_file
        start_time = time.time()
        lines = []
        with open(corpus_file, 'rt') as f:
            lines = f.readlines()

        num = len(lines)
        vector_size = self.fbuilder.feature_size
        X = np.zeros(shape=(num, vector_size))
        Y = np.zeros(shape=(num, 1))

        old_domain = ''
        train_dir_path = self.data_path#os.path.join(self.data_path, 'lett.train') 
        for idx, line in enumerate(lines):
            en_url, fr_url, label = line.split('\t')
            #TODO next try to get feature from cache first then if not exist, then read the corpus and file

            domain = get_domain(en_url)
            if old_domain != domain:
                print '---build feature for:', domain
                en_corpus, fr_corpus = read_lett(os.path.join(train_dir_path, domain + '.lett.gz'), 'en', 'fr')
                old_domain = domain

            en_page = en_corpus[en_url]
            fr_page = fr_corpus[fr_url]
            X[idx] = self.fbuilder.get_features(en_page, fr_page)
            Y[idx] = int(label)

        dataset = DataSet(X, Y)

        print '----------Finish build dataset %s in %d sec'%(corpus_file, time.time - start_time)
        return dataset

                

#================================For testing========================
from ml_app.feature_builder.bigram_counter import BigramCounter

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    dprovider = FullDataProvider(config)
    pdb.set_trace()

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

