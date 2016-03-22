#!/usr/bin/python
#runned on python 2.7 at at Rotunda laboratory

import os, sys, glob, base64, gzip, re
from collections import namedtuple
import fileinput
LENGTH_UPPER_BOUND = 1.3
LENGTH_LOWER_BOUND = 0.7
COMMON_WORDS_BOUND = 0.02
CORPUS_PATH = "/tmp/u/vutrongh/strict/lett.train"

Page = namedtuple(
    "Page", "url, text, lang, length, token_set")


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
            p = Page(url, text, lang, len(text), set(re.split('\W+', text)))
            yield p


def read_lett(f, source_language, target_language):
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter(f):
    
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




sys.stderr.write("loading corpus\n")
sources, targets = read_all_lett(glob.glob(CORPUS_PATH +"/*.lett.gz"), 'en', 'fr')
sys.stderr.write("filtering\n")
for index, line in enumerate(fileinput.input()):
    source_url, target_url = line[:-1].split("\t")
    source = sources[source_url]
    target = targets[target_url]
    lrate = source.length/float(target.length)
    if(lrate > LENGTH_LOWER_BOUND and lrate < LENGTH_UPPER_BOUND):
        score = len(source.token_set.intersection(target.token_set))\
                /float(len(source.token_set))
        if(score > COMMON_WORDS_BOUND):
            print line.strip()
sys.stderr.write("filter done\n")
