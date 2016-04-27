
from __future__ import print_function, division
import os
import math
import sys
import time
import numpy as np
from collections import defaultdict
from contextlib import contextmanager

from app_funs import read_lett, get_domain

#'''
data_path = '../ml_app/data'
train_pairs = os.path.join(data_path, 'train.pairs')
lett_path = os.path.join(data_path, 'lett.train')
tran_en = os.path.join(data_path, 'translations.train/url2text.en')
#'''

'''
data_path = '/tmp/u/vutrongh/lett.train'
train_pairs = '../../../train.pairs'#os.path.join(data_path, 'train.pairs')
lett_path = '/tmp/u/vutrongh/lett.train'# os.path.join(data_path, 'lett.train')
tran_en = '/tmp/u/vutrongh/translations.train/url2text.en'#os.path.join(data_path, 'tiranslations.train/url2text.en')
'''

'''
test_outputs = './test_outputs'
test_debugs = './test_debugs'
data_path = '../ml_app/data/test'
lett_path = os.path.join(data_path, 'lett.test')
tran_en = os.path.join(data_path, 'translations.test/url2text.en.detok')
'''

current_domain = ''
en_corpus = None#dict, Enlgish page of the current domain access by en_corpus[url]
fr_corpus = None#dict, Frech page of the current domain
tran_corpus = None#dict, translatin of all French pages in the current doamin access by
models = None#dict, store all unigram language model of translation for the domain currently processing
col_model = None#unigram language model for all translation of the current domain (the collection)
col_size = None#number of word in the current collection
col_vocab_size = None#vocabulary size of the current collection
lamda = 0.5#0.5 Best#TODO: smooth parameter find the optimal, is it important for thi app?
debug = True
output_top = 10
use_filter = False

def print_err(msg):
    sys.stderr.write(msg + '\n')

@contextmanager
def time_it(msg_in, msg_out):
    print(msg_in)
    sys.stderr.write(msg_in + '\n')
    start = time.time()
    yield
    duration = time.time() - start
    print(msg_out % duration)
    sys.stderr.write(msg_out % duration + '\n')

def load_translation(domain):
    '''Load transation for the given domain, ignore all line of others domain.'''
    global tran_corpus
    tran_corpus = defaultdict(list)
    domain_found = False
    with open(tran_en, 'rt') as f:
        for line in f:
            url, line = line.strip().split('\t')
            url, line = url.strip(), line.strip().lower()
            if url == 'unknown_url': continue
            line_domain = get_domain(url)
            if line_domain == domain:#a domain have transaltion in multiple line
                tran_corpus[url].append(line)
                domain_found = True
            elif domain_found:
                break

def compute_models():
    '''Build all unigram for all translation of the current domain.'''
    global models, col_model, col_size, col_vocab_size
    models = {}
    col_model = defaultdict(lambda : 0.0)
    col_size = 0.0

    for url in tran_corpus:
        models[url] = get_pro_model('\n'.join(tran_corpus[url]))
    col_vocab_size = len(col_model.keys())

def load_domain_corpus(domain):
    '''Load all English page, French page (if the domain is not the current one).'''
    '''each domain will be load only one time to memory'''
    global current_domain, en_corpus, fr_corpus
    if current_domain != domain:
        #if current_domain != '':
            #print_domain_summary(current_domain)
        current_domain = domain
        f = os.path.join(lett_path, domain + '.lett.gz')
        with time_it('=========Loading domain', '---Loading completed in %.2f s'):
            en_corpus, fr_corpus = read_lett(f, 'en', 'fr')
            load_translation(domain)
            compute_models()

def get_tokens(text):
    return text.lower().split()

def get_uni_pro_model(text):
    '''Build unigram language model for the given text.'''
    global col_model, col_size
    pro_model = defaultdict(lambda : 0.0)
    words = get_tokens(text)
    for w in words:
        pro_model[w] +=1
        col_model[w] +=1
        col_size += 1

    return pro_model, len(words)

def get_pro_model(text):
    '''Just call the model builder we wish to used.'''
    return get_uni_pro_model(text)

def score_original(fr_url, words):
    #NOTE for faster we could use a simple smoothing such as 1/V or a small number 10^-4
    if fr_url not in models:#don have its translation
        #print '!!!!dont have transation:', fr_url
        return None

    model, word_len = models[fr_url] 
    #doc_vocab_size = len(model.keys())

    #add_nor = col_vocab_size/float(doc_vocab_size)#a kind of simple smooth

    score = 0.0
    #words = get_tokens(text)#Hoa delete it
    for w in words:
        tc = (col_model[w] + 1.0)/(col_size + col_vocab_size)#term collection score
        td = model[w]/word_len#term translation score
        score += math.log(lamda*td + (1-lamda)*tc)#linear interporation, jelinek-mercer smoothing

        '''Simple smoothing
        if w in model.keys():
            #score += math.log((model[w]+add_nor)/(word_len+col_vocab_size))
            score += math.log((model[w])/(word_len))
        else:#smoothing...
            #add-one
            #score += math.log(1.0/col_vocab_size)
            score += math.log(1.0/col_vocab_size)
        '''

    return score

def col_model_for_a_doc(words, unique_tokens):
    doc_col_model_scores = {}
    #for idx, w in enumerate(words):
    for w in unique_tokens:
        doc_col_model_scores[w]= {'count':words.count(w), 'col_score':(col_model[w] + 1.0)/(col_size + col_vocab_size)}#term collection score
    return doc_col_model_scores

def score_original_optimal(fr_url, words, doc_col_scores, max_score):
    if fr_url not in models:#don have its translation
        return None

    model, word_len = models[fr_url] 

    score = 0.0
    for w in words: #words is unique tokens
        #tc = (col_model[w] + 1.0)/(col_size + col_vocab_size)#term collection score
        w_col_score = doc_col_scores[w]
        td = model[w]/word_len#term translation score
        #score += math.log(lamda*td + (1-lamda)*tc)#linear interporation, jelinek-mercer smoothing
        score += w_col_score['count']*math.log(lamda*td + (1-lamda)*w_col_score['col_score'])#linear interporation, jelinek-mercer smoothing
        if score<max_score:
            return None

    return score



def sort_candidates(cans, scores):
    '''Get two list, cans[0] with a coresponding score at scores[0], then sort them decreaseing by its score.'''
    cans = np.array(cans)
    scores = np.array(scores)
    idxs = np.argsort(scores)[::-1]
    cans = cans[idxs]
    scores = scores[idxs]

    return cans, scores

def find_rank_gold(cans, gold):
    '''Find index of gold in cans. Return None if not found'''
    rank = np.where(cans==gold)[0]
    if rank.shape[0]>0:
        rank = rank[0]
    else:
        rank = None

    return rank

LENGTH_UPPER_BOUND = 2.0
LENGTH_LOWER_BOUND = 0.5
#SHARED_WORDS_THRES = 0.015

def get_candidates(en_url):
    '''Get all candidates for the given source English URL'''
    #domain = get_domain(en_url)#TODO get these coment back for run train data?? doo all en_url of a domain, so load 
    #load_domain_corpus(domain)

    en_page = en_corpus[en_url]
    unique_en_tokens = list(set(en_page.tokens))
    doc_col_scores = col_model_for_a_doc(en_page.tokens, unique_en_tokens)

    cans, scores = [''], [0]
    max_score = float('-inf')
    
    for fr_url in fr_corpus:
        lrate = en_page.length/float(fr_corpus[fr_url].length)
        #if use_filter and (lrate < LENGTH_LOWER_BOUND or lrate > LENGTH_UPPER_BOUND):#filter length
                #continue
        #score = score_original(fr_url, en_page.tokens)
        #print('1:' + str(score))
        score = score_original_optimal(fr_url, unique_en_tokens, doc_col_scores, max_score)
        #print('2:' + str(score))
        if score is not None:
            #cans.append(fr_url)
            #scores.append(score)
            if score>max_score:
                max_score = score
                cans[0] = fr_url
                scores[0] = score

    cans, scores = sort_candidates(cans, scores)
    return cans, scores


######Only for printing the results not nassary to read
count, top1, top5, top10, top2 = 0.0, 0, 0, 0, 0
dcount, dtop1, dtop5, dtop10, dtop2 = 0.0, 0, 0, 0, 0
dtime = time.time()

def print_debug(en_url, cans, scores, gold, rank):
    print('-'*10, en_url)
    print('---gold(rank=%d, score=%.3f):%s'%(rank, scores[rank],gold))
    for idx in range(min(10, cans.shape[0])):
        print('%.3f:\t%s'%(scores[idx], cans[idx]))

def count_evaluate(en_url, cans, scores, gold):
    global count, top1, top5, top10, dcount, dtop1, dtop5, dtop10, top2, dtop2
    count += 1
    dcount += 1

    rank = find_rank_gold(cans, gold)
    if rank == None:
        print('!!!Gold is not in cans!!! Translation is missing')
        print_debug(en_url, cans, scores, gold, -1)
        return

    print_debug(en_url, cans, scores, gold, rank)

    mark = None
    if rank==0:
        top1 +=1
        dtop1 +=1
        if mark is None: mark = 1
    if rank<2:
        top2 += 1
        dtop2 += 1
        if mark is None: mark = 2
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
    global count, top1, top5, top10, dcount, dtop1, dtop5, dtop10, dtime, dtop2
    mess = '---domain: %s in %.2fs: total: %d, top1: %.3f, top2:%.3f, top5: %.3f, top10: %.3f\n'%(domain, time.time() - dtime, dcount, dtop1/dcount, dtop2/dcount, dtop5/dcount, dtop10/dcount)
    print(mess)
    sys.stderr.write(mess)
    dcount, dtop1, dtop2, dtop5, dtop10 = 0.0, 0, 0, 0, 0
    dtime = time.time()

def print_summary():
    mess = '-Summary: total: %d, top1: %.3f, top2: %.3f, top5: %.3f, top10: %.3f\n'%(count, top1/count, top2/count, top5/count, top10/count)
    print(mess)
    sys.stderr.write(mess)

def run1():
    debug_domains = ['www.dakar.com', 'www.luontoportti.com', 'www.nauticnews.com', 'www.the-great-adventure.fr']
    with open(train_pairs, 'rt') as f:
        for line in f:
            en_url, gold = line.strip().split('\t')
            en_url, gold = en_url.strip(), gold.strip()

            if debug and get_domain(en_url) not in debug_domains:
                continue

            cans, scores = get_candidates(en_url)
            count_evaluate(en_url, cans, scores, gold)

    print_domain_summary(current_domain)
    print_summary()

def print_test_debug(en_url, cans, scores):
    print('------en_url_source:%s'%en_url)
    m = min(output_top, cans.shape[0])
    if m<=0:
        return
    for idx in range(m):
        print('%d\t%.3f\t%s'%(idx, scores[idx], cans[idx]))

def predict_one_domain(domain):
    #debug_file = domain + '.debug.txt'
    #output_file = domain + '.output.txt'
    load_domain_corpus(domain)
    for en_url in en_corpus:
        cans, scores = get_candidates(en_url)
        print_test_debug(en_url, cans, scores)

def run2():#get the result to submit
    '''Only print out top 5 each domain for output, top10 for debugs in a separated file, so later if we need fix a domain without transation or something we don't need to rerun every thing again.'''
    #Get lett file in test_data, get candiate for every single en_url available
    #domain results in a file, debut in a file,
    #list all lett file, call predic one domain
    debug_domain_files = ['bugadacargnel.com', 'harasdesfrettes.com']
    for domain in os.listdir(lett_path):
        domain = domain[:domain.find('.lett.gz')]
        if debug and domain  not in debug_domain_files:
            continue
        with time_it('================Processing domain:' + domain, '======domain,' + domain + ', completed in %.2f s'):
           predict_one_domain(domain)
        #break

        

def print_page_content(corpus, url):
    #print content in a reable fomat for corpus[url] 
    pass

def run_debug():
    pass
    #print content and see why it worng

if __name__ == '__main__':
    #run1()
    run2()
