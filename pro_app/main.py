from __future__ import print_function, division
import os
import math
import sys
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

from app_funs import read_lett, get_domain

data_path = '../ml_app/data'
train_pairs = os.path.join(data_path, 'train.pairs')
lett_path = os.path.join(data_path, 'lett.train')
tran_en = os.path.join(data_path, 'translations.train/url2text.en')

current_domain = ''
en_corpus = None
fr_corpus = None
tran_corpus = None
models = None
col_model = None
col_size = None
lamda = 0.5 #TODO: smooth parameter find the optimal, is it important for thi app?


@contextmanager
def time_it(msg_in, msg_out):
    print(msg_in)
    start = time()
    yield
    duration = time() - start
    print(msg_out % duration)

def load_translation(domain):
    tran_corpus = defaultdict(list)
    with open(tran_en, 'rt') as f:
        for line in f:
            url, line = line.strip().split('\t')
            url, line = url.strip(), line.strip().lower()
            line_domain = get_domain(url)
            if line_domain == domain:
                tran_corpus[url].append(line)
                break

def compute_models():
    '''Compute models for all translation of fr documents.'''
    models = {}
    col_model = defaultdict(lambda : 0.0)
    col_size = 0.0

    for url in tran_corpus:
        models[url] = get_pro_model('\n'.join(tran_corpus[url]))

def load_domain_corpus(domain):
    if current_domain != domain:
        if current_domain != '':
            print_domain_summary(current_domain)
        current_domain = domain
        f = os.path.join(lett_path, domain + '.lett.gz')
        print('========Loading domain')
        with time_it('---Loading completed in %.2f ms'):
            en_corpus, fr_corpus = read_lett(f, 'en', 'fr')
            load_translation(domain)
            compute_models()

def get_tokens(text):
    return text.lower().split()

def get_uni_pro_model(text):
    '''Build unigram language model for the given text.'''

    pro_model = defaultdict(lambda : 0.0)
    words = get_tokens(text)
    for w in words:
        pro_model[w] +=1
        col_model[w] +=1
        col_size += 1

    return pro_model, len(words)

def get_pro_model(text):
    return get_uni_pro_model(text)

def score_original(fr_url, text):
    #TODO need smoothing, simple and small <1/V
    if fr_url not in models:#don have its translation
        #print '!!!!dont have transation:', fr_url
        return None

    #'''
    model, word_len = models[fr_url] 
    col_vocab_size = len(col_model.keys())
    doc_vocab_size = len(model.keys())

    add_nor = col_vocab_size/float(doc_vocab_size)
    #'''

    score = 0.0
    words = get_tokens(text)
    for w in words:
        tc = (col_model[w] + 1.0)/(col_size + col_vocab_size)
        td = model[w]/word_len
        score += math.log(lamda*td + (1-lamda)*tc)
        '''
        if w in model.keys():
            #score += math.log((model[w]+add_nor)/(word_len+col_vocab_size))
            score += math.log((model[w])/(word_len))
        else:#smoothing...
            #add-one
            #score += math.log(1.0/col_vocab_size)
            score += math.log(1.0/col_vocab_size)
        '''

    return score

def sort_candidates(cans, scores):
    cans = np.array(cans)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    cans = cans[idxs]
    scores = scores[idxs]

    return cans, scores

def find_rank_gold(cans, gold):
    rank = np.where(cans==gold)[0]
    if rank.shape[0]>0:
        rank = rank[0]
    else:
        rank = None

    return rank


def get_candidates(en_url):
    domain = get_domain(en_url)
    load_domain_corpus(domain)

    en_page = en_corpus[en_url]

    cans, scores = [], []
    for fr_url in fr_corpus:
        score = score_original(fr_url, en_page.text)
        if score is not None:
            cans.append(fr_url)
            scores.append(score)

    cans, scores = sort_candidates(cans, scores)
    return cans, scores


######Only for printing the results not nassary to read

count, top1, top5, top10 = 0.0, 0, 0, 0
dcount, dtop1, dtop5, dtop10 = 0.0, 0, 0, 0

def print_debug(en_url, cans, scores, gold, rank):
    print('-'*10, en_url)
    print('---gold(rank=%d, score=%.3f):%s'%(rank, scores[rank],gold))
    for idx in range(15):
        print('%.3f:\t%s'%(scores[idx], cans[idx]))

def count_evaluate(en_url, cans, scores, gold):
    count += 1
    dcount += 1

    rank = find_rank_gold(cans, gold)
    if rank == None:
        print('!!!Gold is not in cans!!!')
        print_debug(en_url, cans, scores, gold, -1)
        return

    print_debug(en_url, cans, scores, gold, rank)

    mark = None
    if rank==0:
        top1 +=1
        dtop1 +=1
        if mark is None: mark = 1
    if rank<5:
        top5 +=1
        dtop5 +=1
        if mark is None: mark = 5
    if rank<10:
        top10+=1
        dtop10+=1
        if mark is None: mark = 10

    print('-note for debug: Gold in top', mark)

def print_domain_summary(domain):
    mess = '---domain: %s in %.2fs: total: %d, top1: %.3f, top5: %.3f, top10: %.3f\n'%(domain, time.time() - dtime, dcount, dtop1/dcount, dtop5/dcount, dtop10/dcount)
    print(mess)
    sys.stderr.write(mess)
    dcount, dtop1, dtop5, dtop10 = 0.0, 0, 0, 0
    dtime = time.time()

def print_summary():
    mess = '-Summary: total: %d, top1: %.3f, top5: %.3f, top10: %.3f\n'%(count, top1/count, top5/count, top10/count)
    print(mess)
    sys.stderr.write(mess)

def run1():
    #ignore_domain = ['bugadacargnel.com', 'cbsc.ca', 'cineuropa.mobi', 'creationwiki.org', 'eu2007.de']
    #ignore_domain = ['bugadacargnel.com', 'kustu.com']
    domains = ['bugadacargnel.com']
    #domains = ['kustu.com']
    with open(train_pairs, 'rt') as f:
        for line in f:
            en_url, gold = line.strip().split('\t')
            if len(domains)>0 and get_domain(en_url) not in domains:
                continue
            en_url, gold = en_url.strip(), gold.strip()
            cans, scores = get_candidates(en_url)
            count_evaluate(en_url, cans, scores, gold)

    print_domain_summary(current_domain)
    print_summary()

def print_page_content(corpus, url):
    #print content in a reable form for corpus[url] 
    pass

def run_debug():
    pass
    #print content and see why it worng

if __name__ == '__main__':
    run1()
