import os
import numpy as np
import time

import autopath
from ml_app.utils.app_funs import get_domain, read_lett

class DataSet(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self._num_examples = X.shape[0]
        self._index_in_epoch = 0
        self._epochs_completed = 0 

        #TODO, should we scale feature to 0 and 1????

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            perm = np.arange(self.num_examples)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.Y = self.Y[perm]

            start = 0
            self._index_in_epoch = batch_size

            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self.X[start:end], self.Y[start:end]

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
        self.validation = self.build_dataset(self.valid_file)
        self.test = self.build_dataset(self.test_file)

    def _read_dataset():#No, prefer to cache only feature of docudment in feature builder
        pass

    def build_dataset(self, corpus_file):
        start_time = time.time()
        print '----------Build dataset:', corpus_file
        lines = []
        with open(corpus_file, 'rt') as f:
            lines = f.readlines()

        num = len(lines)
        vector_size = self.fbuilder.feature_size
        X = np.zeros(shape=(num, vector_size))
        Y = np.zeros(shape=(num,))

        old_domain = ''
        train_dir_path = self.data_path#os.path.join(self.data_path, 'lett.train') 
        for idx, line in enumerate(lines):
            en_url, fr_url, label = line.split('\t')

            cache_features = self.fbuilder.get_cached_features(en_url, fr_url)
            if cache_features is not None:
                X[idx] = cache_features
                Y[idx] = int(label)
                continue

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

        print '----------Finish build dataset %s in %f sec'%(corpus_file, time.time() - start_time)
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
    import pdb
    pdb.set_trace()

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

