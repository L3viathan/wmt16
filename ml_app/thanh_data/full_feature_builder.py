import numpy as np

vector_size = 6 #TODO: set it, the size of feature for a candidate pair

def build_dataset(corpus_file):
    '''Build full size matrix for a corpus e.g. train/valid/test.
    Eache row is feature for a pair of documents.
    '''
    print '----------Build dataset:', corpus_file
    lines = []
    with open(corpus_file, 'rt') as f:
        lines = f.readlines()

    num = len(lines)
    X = np.zeros(shape=(num, vector_size))
    Y = np.zeros(shape=(num,))

    for idx, line in enumerate(lines):
        en_url, fr_url, label = line.split('\t')

        Y[idx] = int(label)
        features = [0, 1, 2, 3, 4, 5]#TODO: your feature for a candidate pair, en_url, fr_url 
        X[idx] = features

    return X, Y

def get_one_url_corpus(en_url):
    '''Get features for the given en_url page and all fr_url candidate.'''
    fr_candidates = ['http://fr.', 'http://abc.fr']#TODO: list of most potential fr_url candidates for the given en_url

    num = len(fr_candidates)
    X = np.zeros(shape=(num, vector_size))
    fr_urls = []
    for idx, fr_url in enumerate(fr_candidates):

        features = [0, 1, 2, 3, 4, 5]#TODO: your feature for a candidate pair, en_url, fr_url 

        X[idx] = features
        fr_urls.append(fr_url)

    return X, fr_urls

def main():
    corpus_file = '../ml_app/data/valid_enriched10.txt'
    X, Y = build_dataset(corpus_file)

    Xone, fr_urls = get_one_url_corpus('http://ace.en')

    print X.shape, Y.shape, Xone.shape, len(fr_urls)

if __name__ == '__main__': 
    main()

