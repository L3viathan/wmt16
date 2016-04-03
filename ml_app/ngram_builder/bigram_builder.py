import os
import codecs 

import autopath
from ml_app.utils.io_funs import text_file_line_iter, object_to_file, file_to_object, is_file
from ml_app.utils.app_funs import get_domain, read_lett 

class BigramBuilder(object):

    def __init__(self, config):
        self.full_config = config
        self.config = config['ngram_builder']['BigramBuilder']
        self.data_path = self.full_config['general']['data_path']
        self.train_file = self.full_config['general']['train_file']
        self.en_word_standardizer = self.config['en_word_standardizer']
        self.fr_word_standardizer = self.config['fr_word_standardizer']
        self.result_file = self.config['result_file']

        self.source_lang = 'en'
        self.target_lang = 'fr'
        self.en_bigram = {}
        self.fr_bigram = {}
        
        self.count_bigram_from_train()

    def _save_to_file(self):
        bigram = {'source': self.en_bigram, 'target': self.fr_bigram, 'desc': 'source: English, target: French'}
        print '---Writing bigram to file:', self.result_file
        object_to_file(bigram, self.result_file)

    def _read_from_file(self):
        if is_file(self.result_file):
            print '---Reading bigram from file:', self.result_file
            bigram = file_to_object(self.result_file)
            self.en_bigram = bigram['source']
            self.fr_bigram = bigram['target']
            return True
        return False

    def add_bigram_from_content(self, bigram, content, word_standardizer=None):
        content = content.strip()
        w1 = '<begin>'
        for word in content.split():
            if word_standardizer is not None:
                word = word_standardizer(word)
            w2 = word
            if w1 not in bigram:
                bigram[w1] = {}
            if w2 not in bigram[w1]:
                bigram[w1][w2]=1
            else:
                bigram[w1][w2]+=1
            w1 = w2
        w2 = '<end>'
        if w1 not in bigram:
            bigram[w1] = {}
        if w2 not in bigram[w1]:
            bigram[w1][w2] = 1
        else:
            bigram[w1][w2] +=1

        #TODO: is there something need to be coolected for choosing bigram

    def count_bigram_from_train(self):
        if self._read_from_file():
            return
        old_domain = ''
        sites = None
        for pair in text_file_line_iter(self.train_file):
            en_url, fr_url = pair.split('\t')
            en_url = en_url.strip()
            fr_url = fr_url.strip()
            domain = get_domain(en_url)
            if old_domain !=domain:
                old_domain = domain
                print '---Working in domain: ', domain
                source_corpus= None
                target_corpus= None
                source_corpus, target_corpus = read_lett(os.path.join(self.data_path, domain + '.lett.gz'), self.source_lang, self.target_lang)
            #print en_url 
            self.add_bigram_from_content(self.en_bigram, source_corpus[en_url].text, self.en_word_standardizer)
            #print fr_url 
            self.add_bigram_from_content(self.fr_bigram, target_corpus[fr_url].text, self.fr_word_standardizer)
        
        self._save_to_file()

    def export_to_csv(self, fname, lang):
        bigram = self.en_bigram
        if lang=='fr':
            bigram = self.fr_bigram
        
        with codecs.open(fname, 'wt', encoding='utf8') as f:
            for w1 in bigram:
                for w2 in bigram[w1]:
                    f.write('%s\t%s\t%d\n'%(w1, w2, bigram[w1][w2]))

    def _inverse_count_bigram(self, bigram):
        inv_bigram = {}
        for w1 in bigram:
            for w2 in bigram[w1]:
                count = bigram[w1][w2]
                if count not in inv_bigram:
                    inv_bigram[count] = []
                inv_bigram[count].append((w1, w2))
        return inv_bigram

    def get_top_bigram(self, lang, occur_above=None, top_n=None):
        bigram = self.en_bigram
        if lang=='fr':
            bigram = self.fr_bigram

        inv_bigram = self._inverse_count_bigram(bigram)
        count = 0 
        new_bigram = {}
        for order in sorted(inv_bigram.keys(), reverse=True):
            if occur_above is not None and order<occur_above:
                return new_bigram, count

            #print '----order=%d, len=%d'%(order, len(inv_bigram[order])), inv_bigram[order][0:10]

            for w1, w2 in inv_bigram[order]:
                if w1 not in new_bigram:
                    new_bigram[w1] = {}
                new_bigram[w1][w2] = order
                count += 1
                if top_n is not None and count >= top_n:
                    return new_bigram, count

        return new_bigram, count#couldn't get enough topn

#================================For testing========================
config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    bigram = BigramBuilder(config)
    #bigram.count_bigram_from_train()
    #bigram.export_to_csv('en_bigram.tsv', 'en')
    #bigram.export_to_csv('fr_bigram.tsv', 'fr')
    occur_above = 10
    en_bigram, count = bigram.get_top_bigram('en', occur_above)
    print '---Total en_bigram used: ', count
    fr_bigram, count = bigram.get_top_bigram('fr', occur_above)
    print '---Total fr_bigram used: ', count

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()
