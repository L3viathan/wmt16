import random
import codecs

import autopath
from ml_app.utils.io_funs import text_file_line_iter
from ml_app.utils.app_funs import get_domain, read_lett

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def get_corpus_by_domain(data_path):
    pass

def add_negative_examples(corpus_set, en_url, fr_url, fr_corpus, top_n=None):
    count = 0
    fr_urls = fr_corpus.keys()
    if top_n>len(fr_urls):
        top_n = len(fr_urls)
    idxs = random.sample(range(len(fr_urls)), top_n)
    for idx in idxs:
        #url = fr_urls[idx].decode('utf-8')#some url includes utf-8
        url = fr_urls[idx]
        if url != fr_url:
            line = '%s\t%s\t0'%(en_url, url)
            corpus_set.append(line)
            count +=1
            if top_n is not None and count>top_n:
                break

    return count

def enrich_data(corpus_file, out_file, top_n=100):
    source_lang = 'en'
    target_lang = 'fr'
    data_path = '../data/'
    train_dir_path = data_path + 'lett.train/'
    corpus_set = []
    old_domain = ''
    positive_count = 0
    negative_count = 0
    for line in text_file_line_iter(data_path + corpus_file):
        line = line.strip()
        corpus_set.append(line + '\t1')
        en_url, fr_url = line.split('\t')
        en_url = en_url.strip()
        fr_url = fr_url.strip()
        domain = get_domain(en_url)
        if old_domain != domain:
            if old_domain != '':
                print 'Positive example: %d, negative example: %d'%(positive_count, negative_count)
            print '---enrich: Working in domain:', domain
            positive_count = 0
            negative_count = 0
            source_corpus = None
            target_corpus = None
            source_corpus, target_corpus = read_lett(train_dir_path + domain + '.lett.gz', source_lang, target_lang)
            old_domain = domain
        nega_count = add_negative_examples(corpus_set, en_url, fr_url, target_corpus, top_n=top_n)
        #print en_url, ':', nega_count
        positive_count += 1
        negative_count += nega_count

    #'''
    with open(out_file, 'wt') as f:
        f.write('\n'.join(corpus_set))
    '''
    with codecs.open(out_file, 'wt', encoding='utf-8') as f:
        f.write('\n'.join(corpus_set))
    '''

def main():
    get_config()
    corpus_file = 'train.txt'
    corpus_file = 'valid.txt'
    corpus_file = 'test.txt'
    negative_example = 10
    out_file = 'train_enriched%d.txt'%negative_example
    out_file = 'valid_enriched%d.txt'%negative_example
    out_file = 'test_enriched%d.txt'%negative_example
    enrich_data(corpus_file, out_file, negative_example)

if __name__ == '__main__':
    main()



