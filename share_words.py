#!/usr/bin/python
#runned on python 2.7 at at Rotunda laboratory
# ./lexical_filter.py train.pairs CORPUS_PATH

import os, sys, glob, base64, gzip, re
from collections import namedtuple, defaultdict
import fileinput, threading, Queue
#from nltk.tokenize import sent_tokenize, word_tokenize
LENGTH_UPPER_BOUND = 1.7
LENGTH_LOWER_BOUND = 0.6
SHARED_WORDS_THRES = 0.016
SHARED_POSITION_AWARE_THRES = 0.18

CORPUS_PATH = "/tmp/u/vutrongh/lett.train/"

Page = namedtuple(
    "Page", "url, lang, length, tokens")


#read a dictionary file
def init(dict_file):
    global fr_en_dict
    fr_en_dict = {}
    with open(dict_file,'r') as f:
        for line in f:
            try:
                fr, en = line[:-1].split()
                if(len(fr) < 2): continue # I dont like 1-letter words
                fr_en_dict[fr] = en
            except:
                sys.stderr.write(line+"\n")



#words containing numbers or uppercase letters
#They are called constant
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
    for sent in sent_tokenize(text):
        tokens = word_tokenize(sent)
        ts.extend(tokens)
        flag = False
        for token in tokens:
            if((not flag) and is_word(token)):
                flag = True
                continue
            if (flag and is_constant(token)):
                cs.append(token)
    return ts, cs


#do not use NLTK as it much more slower
def get_constants_nonltk(text):
    ts, cs = [], []
    ts = re.split("\W+", text.lower())
    #ts = text.lower().split()
    #for token in ts:
        #if(is_constant(token)):
            #cs.append(token)
    return ts, cs



# stolen from provided script
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
            tokens, constants = get_constants(text)
            p = Page(url, text, lang, len(tokens), tokens, constants)
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
        p = Page(url, lang, len(tokens), tokens)
        yield p


def read_lett(f, source_language, target_language):
    source_corpus, target_corpus = {}, {}
    sys.stderr.write("loading " + f.name + "...\n")
    for p in read_lett_iter(f): 
        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    sys.stderr.write("finished loading \n")
    return source_corpus, target_corpus

def read_lett_nonltk(f, source_language, target_language):
    sys.stderr.write("loading " +str(f) + "...\n")
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter_nonltk(open (f, 'r')):

        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    return source_corpus, target_corpus



# read all lett files into memory
def read_all_lett(files, source_lang, target_lang):
    source_corpus, target_corpus = {}, {}
    for file in files:
        sys.stderr.write("loading " + file + "...\n")
        for p in read_lett_iter(open(file, 'r')):
            if p.lang == source_lang:
                source_corpus[p.url] = p 
            elif p.lang == target_lang:
                target_corpus[p.url] = p 
            else:  # ignore all other languages
                pass
    return source_corpus, target_corpus



# extract positioned intersection between 2 list token 
# run test_doc_inter to see how it works
# 
# First all tokens1 will be put in a map called map1 
# then checking all in tokens2 wheather they are in map 1 => intersection
# Howerver some tokens may appear more than once. 
# We simply assign first token in source with first in target and so on ....
# occur_maps are used to track this 
def doc_inter(tokens1, tokens2, cached_occur_map1, cached_map1):
    idx1, idx2 = 0, 0
    occur_map1, map1 = cached_occur_map1, cached_map1
    if(not occur_map1 or not map1): # no cache
        occur_map1 = defaultdict(lambda:0)
        map1, idx1, idx2 = {}, 0, 0;
        for token in tokens1:
            if (token in map1):
                occur = occur_map1[token]
                map1[token+"__" + str(occur)] = idx1
            else:
                map1[token] = idx1
            occur_map1[token] = occur_map1[token] + 1
            idx1 = idx1 + 1


    occur_map2 = defaultdict(lambda:0)
    inter=[]
    for token in tokens2:

        lower = token.lower()# use dictionary here
        if(not (token in map1) and (lower in fr_en_dict)):
            token = fr_en_dict[lower] # convert this to english
        if (token in map1):
            if (token not in occur_map2):
                inter.append([token, map1[token], idx2])
            elif(token in occur_map2 and (token +"__"+ str(occur_map2[token]) in map1)):# already occur
                inter.append([token, map1[token +"__"+ str(occur_map2[token])], idx2])
        occur_map2[token] = occur_map2[token] + 1
        idx2 = idx2 + 1
    return inter, occur_map1, map1


def test_doc_inter():
    sent1 = "Anna is a girl on 2nd floor . Thanh loves Anna . Anna seems to care about Thanh . They hardly talk to each other"
    sent2 = "Anna study AI . She lives on 2nd floor . She is sweet . She is always smiling like someone gave her 10k Czech crown on the morning. So is Thanh . They are perfect couple . "
    inter, cached_occur_map1, cached_map1 = doc_inter(sent1.split(), sent2.split(), None, None)
    print sent1
    print sent2
    print inter




# Input: positioned intersection from above function, length of 2 docs
# Simply sum over all difference between each share words.
# but I want the more they are close, the higher score. Then
# I take the negative value by subtracting them by max_length.
def doc_similarity_pos_aware(inter, length1, length2):
    max_len = length1
    sum = 0
    if(len(inter) == 0):
        return 0
    for p in inter:
        sum = sum + (max_len - abs(p[2] -p[1]))/float(max_len)
    return sum


def read_translation  (fn):
    trans = {}
    with open(fn, 'r') as f:
        for line in f:
            url, text = line[:-1].split("\t")
            text = base64.b64decode(text)
            trans[url] = re.split("\W+", text)
    return trans
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
    if(score < 0.0001 and score > - 0.0001):
        return len(item2[0]) - len(item1[0])
    if(score > 0) : return 1
    return -1

class subThread(threading.Thread):
    def __init__(self, sources, targets, name):
        threading.Thread.__init__(self)
        self.sources = sources
        self.targets = targets
        self.queue = Queue.Queue()
        self.name = name
        self.running = True

    def put_url(self, url):
        sys.stderr.write(self.name + " got new url: " + url +"\n")
        self.queue.put(url, block=True)

def get_runned_urls ():
    path = './'
    fname_start = 'test.result'
    urls = set()
    for f in os.listdir(path):
        if f.startswith(fname_start):
            print_err('-*read result file: ' + f)
            print('-*read result file: ' + f)
            with open(os.path.join(path, f), 'rt') as fresult:
                for line in fresult:
              	    url = line.strip().split("\t")[0]
                    urls.add(url)
    return urls
    

def run(urls, sources, targets, sidx, eidx):
    runned_urls = get_runned_urls()
    for url in urls[sidx:eidx]:
        if url in runned_urls: continue
        source = sources[url]
        occur_map1, map1 = None, None
        candis = {} # possible candidates for 1 url
        for turl in targets:
            target = targets[turl]
            lrate = source.length/float(target.length)
            # filter by length
            if(lrate > LENGTH_LOWER_BOUND and lrate < LENGTH_UPPER_BOUND):
                inter, occur_map1, map1 = doc_inter(source.tokens, target.tokens, occur_map1, map1)
                sim = doc_similarity_pos_aware(inter, source.length, target.length)
                # calculating positioned intersections
                if (sim/source.length > SHARED_POSITION_AWARE_THRES):
                    candis[turl] = sim
        if(len(candis) == 0):
            continue
        # rank all candidates left, take the highest-score one
        #sorted_cans = sorted(candis.items(), cmp:custome_cmp, reverse=True)
        sorted_cans = sorted(candis.items(), cmp=custome_cmp, reverse=True)
	count = -1
	while (count < 10 and count < len(sorted_cans) -1):
            count = count + 1
            turl = sorted_cans[count][0]
            sys.stdout.write(url +"\t" + turl + "\t" + str(sorted_cans[count][1]) + "\t" + str(sorted_cans[count][1]/source.length) + "\n")
        #sys.stdout.write(url +"\t" + sorted_cans[0][0]+ "\t" + str(sorted_cans[0][1])+ "\t" + str(sorted_cans[0][1]/source.length) + "\n")
    return
    #sys.stderr.write(self.name + " finished\n")
	


if __name__ == '__main__': 
    import operator, argparse, math
    from multiprocessing import Process
    # read dictionary file
    init("test_dict.txt")

    parser = argparse.ArgumentParser(description='Runing first exercise of NPFL103')
    parser.add_argument('-b', metavar='begin_at_line', dest='begin', help='Begin process at line', type=int, default=None)
    parser.add_argument('-e', metavar='end_at_line', dest='end', help='End process at line', type=int, default=None)

    args = parser.parse_args()
    domains = read_domains(args.begin, args.end)
 
    num_pro=6
    for fn in domains:
        sources, targets = read_lett_nonltk(CORPUS_PATH + fn, 'en', 'fr')
        urls = sources.keys()
	sys.stderr.write("done reading " + fn + "\n")
	pros = []
        step = len(urls)/float(num_pro)
        for i in range(num_pro):
            sidx = math.floor(i * step)
            eidx = math.floor((i+1)*step)
            if(eidx > len(urls) -1): eidx = len(urls) -1
            sys.stderr.write("from %d to %d\n"%(sidx, eidx))
            process = Process(target=run, args=(urls,sources, targets, int(sidx), int(eidx)))
            pros.append(process)
            process.start()
        for i in range(num_pro):
            pros[i].join()
	
'''
            occur_map1, map1 = None, None
            source = sources[url]
            candis = {} # possible candidates for 1 url
            for turl in targets:
                target = targets[turl]
                lrate = source.length/float(target.length)
                # filter by length
                if(lrate > LENGTH_LOWER_BOUND and lrate < LENGTH_UPPER_BOUND):
                    inter, occur_map1, map1 = doc_inter(source.tokens, target.tokens, occur_map1, map1)
                    sim = doc_similarity_pos_aware(inter, source.length, target.length)
                    # calculating positioned intersections
                    if (sim/source.length > SHARED_POSITION_AWARE_THRES):
                        candis[turl] = sim
            if(len(candis) == 0):
                continue
            # rank all candidates left, take the highest-score one
            sorted_cans = sorted(candis.items(), key = operator.itemgetter(1), reverse=True)
	    count = -1
	    #while (count < 10 and count < len(sorted_cans) -1):
             #   is_target = 0
            #	count = count + 1
	    #	turl = sorted_cans[count][0]
             #   if (train[url] == turl) : is_target = 1
              #  print(url +"\t" + turl + "\t" + str(sorted_cans[count][1]) + "\t" + str(sorted_cans[count][1]/source.length) + "\t" +str(is_target))
            #count = -1
            print(url +"\t" + sorted_cans[0][0]  + "\t" + str(sorted_cans[0][1])+ "\t" + str(sorted_cans[0][1]/source.length))
'''
