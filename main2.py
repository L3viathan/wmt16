from config2 import config

fake = True

Page = namedtuple("Page", "url, html, text, mime_type, encoding, lang")

def read_gold(filename):
    """Return pairs (english, french) of URLs that are gold pairs."""
    gold = {}
    with open(filename) as f:
        for line in f:
            source, target = line.strip().split("\t")
            gold[source] = target
    return gold

def get_domain(url):
    '''Extract the domain of the given url.'''
    pass

def read_source_urls(filename):
    '''Read all ulrs we need to find the translated documents.'''

    if fake: return ['abc.en', 'def.en']

    with open(filename, 'rt') as f:
        urls = f.readline()
    
    return urls

last_domain = None
last_source = None
last_target = None
def read_lett(f, source_language, target_language):
    '''Read all document of source/target language to two separted dict, with keys are urls.'''
    global last_domain, last_source, last_target
    if last_domain is not None and last_domain == f:
        return last_source, last_target

    if not isinstance(f, file):
        f = open(f, 'r')
    source_corpus, target_corpus = {}, {}
    for p in read_lett_iter(f):

        if p.lang == source_language:
            source_corpus[p.url] = p
        elif p.lang == target_language:
            target_corpus[p.url] = p
        else:  # ignore all other languages
            pass
    last_souce = source_corpus#cache the lettfile just read
    last_target = target_corpus
    return source_corpus, target_corpus

def get_domain_corpus(en_url):
    '''Return all data of the domain in given url.'''
    if fake: return None, {Page('abc')}

    domain= get_domain(en_url)
    lett_file = ''
    return return read_lett(lett_file, 'en', 'rf')
    
def execute_pipeline(en_url):
    '''Runing compuation step.'''
    en_corpus, fr_corpus = get_domain_corpus(en_url)

    steps = config['pipeline']

    en_page = en_corpus[en_url]
    candidates = fr_corpus#all target pages of this domain
    scores = np.ndarray()#alwasy score all score of each reaming target page
    
    for op in steps:#NOTE: Only this point is different which provide scores of previous steps to the current step, and don't disguinstic between filter/scorer/ranker
        candidates, scores  = op(en_page, candidates, scores)#op could be filter or scorer, ranker, take look at simple for detail

    return candidates, scores #best candidates and score alreay ranked (ordered) by the last ranker
        
def predict(urls):
    '''Predict the translated urls of the url set.'''
    results = {}    
    for en_url in urls:
        results[en_url] = execute_pipeline(en_url)
    
    return results
        
def evalute(urls, gold, predicts)
    pass

def main():
    source_url_file = config['general']['source_urls']#the urls we required to provide its translateion
    #the urls is listed such that same domain urls is near by together, so we don't need read lett file again and again
    urls = read_source_urls(source_url_file)

    resutls = predict(urls)

    gold_file = config['general']['gold_file']
    gold = read_gold(gold_file)
    top1, top5, top10 = evaluate(urls, gold, predicts) 

if __name__ == '__main__':
    main()

