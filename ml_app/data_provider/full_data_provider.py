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

        #TODO, should we scale feature to 0 and 1????, not this app

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

    def next_balanced_batch(self, batch_size):
        #NOTE: batch size no larger than double the number of possitive/negative examples and also only for binary classification
        batch_x, batch_y = self.next_batch(batch_size)
        negative_idxs = np.where(batch_y==0)[0]
        add_positive_num = len(negative_idxs)-batch_size/2

        reduce_idxs = negative_idxs
        reduce_num = add_positive_num
        reduce_label = 0
        if add_positive_num<0:
            reduce_idxs = np.where(batch_y==1)[0]
            reduce_num = abs(add_positive_num)
            reduce_label = 1

        add_label = abs(1-reduce_label)
        reduce_idxs = np.random.choice(reduce_idxs, reduce_num, replace=False)
        add_idxs = np.where(self.Y==add_label)[0]
        add_idxs = np.random.choice(add_idxs, reduce_num, replace=False)

        batch_x[reduce_idxs] = self.X[add_idxs]
        batch_y[reduce_idxs] = add_label

        return batch_x, batch_y
        

class FullDataProvider(object):
    '''Everything in memory.'''

    def __init__(self, config):
        self.full_config = config
        self.config = config['data_provider']['FullDataProvider']
        self.fbuilder = self.config['feature_builder'](self.full_config)

        #TODO: fake data for fast process of checking everythin fit in together
    
        self.data_path = self.full_config['general']['data_path']
        self.train_dir_path = self.data_path#os.path.join(self.data_path, 'lett.train') 
        self._vector_size = self.fbuilder.feature_size
        self.train_file = self.config['train_file']
        self.valid_file = self.config['valid_file']
        self.test_file = self.config['test_file']

        self.train = self.build_dataset(self.train_file)
        self.validation = self.build_dataset(self.valid_file)
        self.test = self.build_dataset(self.test_file)
        self.fbuilder.save_features()

        self.current_evaluate_domain = ''
        self.current_evaluate_en_corpus = None
        self.current_evaluate_fr_corpus = None

    def build_dataset(self, corpus_file):
        start_time = time.time()
        print '----------Build dataset:', corpus_file
        lines = []
        with open(corpus_file, 'rt') as f:
            lines = f.readlines()

        num = len(lines)
        X = np.zeros(shape=(num, self._vector_size))
        Y = np.zeros(shape=(num,))

        old_domain = ''
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
                en_corpus, fr_corpus = read_lett(os.path.join(self.train_dir_path, domain + '.lett.gz'), 'en', 'fr')
                old_domain = domain

            en_page = en_corpus[en_url]
            fr_page = fr_corpus[fr_url]
            X[idx] = self.fbuilder.get_features(en_page, fr_page)
            Y[idx] = int(label)

        dataset = DataSet(X, Y)

        print '----------Finish build dataset %s in %f sec'%(corpus_file, time.time() - start_time)
        return dataset

    def get_one_url_corpus(self, en_url):
        #TODO: should change to cache only urls of a domain, instead of read lett.gz file again here
        domain = get_domain(en_url)
        if domain != self.current_evaluate_domain:
            en_corpus, fr_corpus = read_lett(os.path.join(self.train_dir_path, domain + '.lett.gz'), 'en', 'fr')
            self.current_evaluate_en_corpus = en_corpus
            self.current_evaluate_fr_corpus = fr_corpus
        else:
            en_corpus = self.current_evaluate_en_corpus
            fr_corpus = self.current_evaluate_fr_corpus
        
        num = len(fr_corpus.keys())
        X = np.zeros(shape=(num, self._vector_size))
        fr_urls = [] 
        for idx, fr_url in enumerate(fr_corpus.keys()):
            features = self.fbuilder.get_cached_features(en_url, fr_url)
            if features is None:
                en_page = en_corpus[en_url]
                fr_page = fr_corpus[fr_url]
                features = self.fbuilder.get_features(en_page, fr_page)
            X[idx] = features 
            fr_urls.append(fr_url)

        return X, fr_urls

#================================For testing========================
from ml_app.feature_builder.bigram_counter import BigramCounter

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    dprovider = FullDataProvider(config)
    #x, y = dprovider.test.next_balanced_batch(350)
    x, urls = dprovider.get_one_url_corpus('http://bugadacargnel.com/en/pages/artistes.php?name=annikalarsson')
    import pdb
    pdb.set_trace()

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

