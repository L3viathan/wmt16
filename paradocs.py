#!/usr/bin/python
#runned on python 2.7 at at Rotunda laboratory
# ./lexical_filter.py train.pairs CORPUS_PATH

import os, sys, glob, base64, gzip, re
from collections import namedtuple, defaultdict
import fileinput, threading, Queue, random
import numpy as np
#from nltk.tokenize import sent_tokenize, word_tokenize
LENGTH_UPPER_BOUND = 1.5
LENGTH_LOWER_BOUND = 0.5
SHARED_WORDS_THRES = 0.0025

CORPUS_PATH = "/tmp/u/vutrongh/data/lett.train/"

Page = namedtuple(
    "Page", "url, lang, length, tokens")

def get_hapax(doc):
    hapax = []
    occurs = defaultdict(lambda:0)
    tokens = doc.split()
    # find duplicate tokens
    for token in tokens:
        occurs[token] = occurs[token] + 1
    dups = {k:v for (k,v) in occurs.items() if v > 1}
    for token in tokens:
        if token in dups: continue
        hapax.extend(token)
    return hapax


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
        tokens = get_hapax(text)
        p = Page(url, lang, len(tokens), tokens)
        yield p

def read_lett(f, source_language, target_language):
    sys.stderr.write("loading " +str(f) + "...\n")
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter(open (f, 'r')):

        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    return source_corpus, target_corpus


def read_domains(begin=None, end=None):
    domains = []
    with open('domains.txt', 'r') as f:
        for idx, line in enumerate(f.readlines()):
            line = line.strip()
            if begin is None or (idx>=begin and idx<end):
                sys.stderr.write('%d: %s\n'%(idx, line))
                domains.append(line)
    return domains

def custome_cmp (item1, item2):
    score = item1[1] - item2[1]
    if (score == 0): return 0;
    if(score > 0) : return 1

    return -1

def get_runned_urls ():
    path = './'
    fname_start = sys.argv[0] + ".result.top10.corrected"
    urls = set()
    for f in os.listdir(path):
        if f.startswith(fname_start[2:]):
            #print_err('-*read result file: ' + f)
            sys.stderr.write('-*read result file: ' + f + "\n")
            with open(os.path.join(path, f), 'rt') as fresult:
                for line in fresult:
              	    url = line.strip().split("\t")[0]
                    urls.add(url)
    return urls



## 
##
## Source code from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
##
def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


    
def output (message):
    with open(sys.argv[0] + ".result.top10", "a") as out_file:
        out_file.write(message)




def run(urls, sources, targets, sidx, eidx, lock):
    runned_urls = get_runned_urls()
    for url in urls[sidx:eidx]:
        if url in runned_urls: continue
        if url not in sources: continue
        source = sources[url]
        candis = {} # possible candidates for 1 url
        for turl in targets:
            target = targets[turl]
            #filter by length
            if (target.length == 0): continue
            lrate = source.length/float(target.length)
            if(lrate > LENGTH_LOWER_BOUND and lrate < LENGTH_UPPER_BOUND):
                #filter by intersection
                inter = set(source.tokens).intersection(target.tokens)
                inter_ = len(inter)/float(max(source.length, target.length))
                if (inter_ > SHARED_WORDS_THRES):
                    sim = levenshtein(target.tokens, source.tokens)
                    candis[turl] = sim/float(max(source.length, target.length))
        if(len(candis) == 0):
            continue
        # rank all candidates left, take the highest-score one
        sorted_cans = sorted(candis.items(), cmp=custome_cmp, reverse=False)
	count = -1
	while (count < 10 and count < len(sorted_cans) -1):
            count = count + 1
            turl = sorted_cans[count][0]
            lock.acquire()
            try:
                output(url +"\t" + turl + "\t" + str(sorted_cans[count][1]) + "\t" + str(sorted_cans[count][1]/source.length) + "\n")
            finally:
                lock.release()
    return
	


if __name__ == '__main__': 
    import operator, argparse, math, time
    from multiprocessing import Process, Lock
    # read dictionary file
    train = {}
    for line in open("train.pairs", 'r'):
        train[line[:-1].split("\t")[0]] = line[:-1].split("\t")[1]

    parser = argparse.ArgumentParser(description='Runing first exercise of NPFL103')
    parser.add_argument('-b', metavar='begin_at_line', dest='begin', help='Begin process at line', type=int, default=None)
    parser.add_argument('-e', metavar='end_at_line', dest='end', help='End process at line', type=int, default=None)

    args = parser.parse_args()
    domains = read_domains(args.begin, args.end)
 
    num_pro=7
    for fn in domains:
        lock = Lock()
       # out_file = open(sys.argv[0] + ".result", "w")
        sources, targets = read_lett(CORPUS_PATH + fn, 'en', 'fr')
        urls = train.keys()
	sys.stderr.write("done reading " + fn + "\n")
	pros = []
        step = len(urls)/float(num_pro)
       	for i in range(num_pro):
            sidx = math.floor(i * step)
            eidx = math.floor((i+1)*step)
            if(eidx > len(urls) -1): eidx = len(urls) -1
            sys.stderr.write("from %d to %d\n"%(sidx, eidx))
            process = Process(target=run, args=(urls,sources, targets, int(sidx), int(eidx), lock))
            pros.append(process)
            process.start()
        for i in range(num_pro):
            pros[i].join() 
