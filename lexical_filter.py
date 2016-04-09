#!/usr/bin/python
#runned on python 2.7 at at Rotunda laboratory

import os, sys, glob, base64, gzip, re
from collections import namedtuple, defaultdict
import fileinput
from nltk.tokenize import sent_tokenize, word_tokenize
LENGTH_UPPER_BOUND = 1.3
LENGTH_LOWER_BOUND = 0.7
COMMON_WORDS_BOUND = 0.02
CORPUS_PATH = "/tmp/u/vutrongh/lett.train"

Page = namedtuple(
    "Page", "url, text, lang, length, tokens, constants, sent_length")

def is_constant (token):
    for i in range(len(token)):
        order = ord(token[i])
        if(64 < order and order < 90) or (48 < order and order < 57):
            return True
def is_word(token):
    found = re.match("\w+", token)
    if found:
        return True
    return False
def get_constants (text):
    ts, cs = [], []
    sent_length = 0
    for sent in sent_tokenize(text):
        sent_length = sent_length + 1
        tokens = word_tokenize(sent)
        ts.extend(tokens)
        flag = False
        for token in tokens:
            if((not flag) and is_word(token)):
                flag = True
                continue
            if (flag and is_constant(token)):
                cs.append(token)
    return ts, cs, sent_length

def get_constants_nonltk(text):
    ts, cs = [], []
    ts = re.split("\W+", text)
    for token in ts:
        if(is_constant(token)):
            cs.append(token)
    return ts, cs


def read_lett_iter(f, decode=True):
    fh = f
    if f.name.endswith('.gz'):
        fh = gzip.GzipFile(fileobj=fh, mode='r')
        for line in fh:
            lang, mine, enc, url, html, text = line[:-1].split("\t")
            #html = base64.b64decode(html)
            text = base64.b64decode(text)

            if decode:
                #html = html.decode("utf-8")
                text = text.decode("utf-8")
            tokens, constants, sent_length = get_constants(text)
            p = Page(url, text, lang, len(tokens), tokens, constants, sent_length)
            yield p
def read_lett_iter_nonltk(f, decode=True):
    fh = f
    if f.name.endswith('.gz'):
        fh = gzip.GzipFile(fileobj=fh, mode='r')
    for line in fh:
        lang, mine, enc, url, html, text = line[:-1].split("\t")

        #html = base64.b64decode(html)
        text = base64.b64decode(text)

        if decode:
           #html = html.decode("utf-8")
            text = text.decode("utf-8")
        tokens, constants = get_constants_nonltk(text)
        p = Page(url, text, lang, len(tokens), tokens, constants)
        yield p


def read_lett(f, source_language, target_language):
    try:
        f.name
    except:
        f = open(f,'r')
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter(f):
    
        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    return source_corpus, target_corpus

def read_lett_nonltk(f, source_language, target_language):
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter_nonltk(f):

        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    return source_corpus, target_corpus



def read_all_lett(files, source_lang, target_lang):
    source_corpus, target_corpus = {}, {}
    for file in files:
        for p in read_lett_iter(open(file, 'r')):
            if p.lang == source_lang:
                source_corpus[p.url] = p 
            elif p.lang == target_lang:
                target_corpus[p.url] = p 
            else:  # ignore all other languages
                pass
    return source_corpus, target_corpus



def doc_inter(tokens1, tokens2, occur_map1, map1):
    idx1, idx2 = -1, 0
    if(not occur_map1 or not map1):
        occur_map1 = defaultdict(lambda:0)
        map1, idx1, idx2 = {}, -1, 0;
        for token in tokens1:
            idx1 = idx1 + 1
            if (token in map1):
                occur = occur_map1[token]
                map1[token+"__" + str(occur)] = idx1
            else:
                map1[token] = idx1
            occur_map1[token] = occur_map1[token] + 1
    occur_map2 = defaultdict(lambda:0)
    inter=[]
    for token in tokens2:
        if (token in map1):
            if (token not in occur_map2):
                inter.append([token, map1[token], idx2])
            elif(token in occur_map2 and (token +"__"+ str(occur_map2[token]) in map1)):# already occur
                inter.append([token, map1[token +"__"+ str(occur_map2[token])], idx2])
        occur_map2[token] = occur_map2[token] + 1
        idx2 = idx2 + 1
    return inter, occur_map1, map1


def doc_similarity_pos_aware(inter, length1, length2):
    max_len = max(length1, length2)
    sum = 0
    if(len(inter) == 0):
        return 0
    for p in inter:
        sum = sum + (max_len - abs(p[2] -p[1]))/float(max_len)
    return sum

if __name__ == '__main__':     
    sys.stderr.write("loading corpus\n")
    sources, targets = read_all_lett(glob.glob(CORPUS_PATH +"/*.lett.gz"), 'en', 'fr')
    sys.stderr.write("filtering\n")
    for index, line in enumerate(fileinput.input()):
        source_url, target_url = line[:-1].split("\t")
        source = sources[source_url]
        target = targets[target_url]
        lrate = source.length/float(target.length)
        if(lrate > LENGTH_LOWER_BOUND and lrate < LENGTH_UPPER_BOUND):
            score = len(source.tokens.intersection(target.tokens))\
                    /float(source.length)
            if(score > COMMON_WORDS_BOUND):
                print line.strip()
    sys.stderr.write("filter done\n")
