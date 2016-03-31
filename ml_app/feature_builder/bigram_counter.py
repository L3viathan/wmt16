import autopath
from ml_app.utils.io_funs import text_file_line_iter, object_to_file, file_to_object, is_file
#from ml_app.ngram_builder.bigram_builder import BigramBuilder
from ml_app.utils.app_funs import Page

class BigramCounter(object):
    
    def __init__(self, config):
        self.full_config = config
        self.config = config['feature_builder']['BigramCounter']
        self.en_word_standardizer = self.config['en_word_standardizer']
        self.fr_word_standardizer = self.config['fr_word_standardizer']
        self._get_extra_features = self.config['add_extra_features']
        self.cache_file = self.config['cache_file']
        self.bigram_filter_level = self.config['bigram_filter_level']

        self.bigram_builder = self.config['bigram_builder'](self.full_config)
        self.en_bigram, _ = self.bigram_builder.get_top_bigram('en', occur_above=self.bigram_filter_level)
        self.fr_bigram, _ = self.bigram_builder.get_top_bigram('fr', occur_above=self.bigram_filter_level)

        self.en_bigram_to_index, self.en_vector_size = self._format_vector(self.en_bigram)
        self.fr_bigram_to_index, self.fr_vector_size = self._format_vector(self.fr_bigram)

        self.source_lang = 'en'
        self.target_lang = 'fr'
        self.en_cache_features = {}
        self.fr_cache_features = {}
        self._read_from_file()
        self.cache_changed = False

    @property
    def feature_size(self):
        return self.en_vector_size + self.fr_vector_size

    def _save_to_file(self):
        cache_features = {'source': self.en_cache_features, 'target': self.fr_cache_features, 'desc': 'page features, source: English, target: French'}
        print '---Writing features to file:', self.cache_file
        object_to_file(cache_features, self.cache_file)

    def _read_from_file(self):
        if is_file(self.cache_file):
            print '---WARNING: Reading features from cache file:', self.cache_file
            cache_features = file_to_object(self.cache_file)
            self.en_cache_features = cache_features['source']
            self.fr_cache_features = cache_features['target']
            return True
        return False

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cache_changed:
            self._save_to_file()
        #super(BigramCounter, self).__exit__(exc_type, exc_value, tracback)
    
    def __del__(self):
        if self.cache_changed:
            self._save_to_file()
        #super(BigramCounter, self).__del__()

    def _format_vector(self, bigram):
        to_index = {}
        count = 0
        for w1 in bigram:
            to_index[w1] = {}
            for w2 in bigram[w1]:
                to_index[w1][w2] = count
                count +=1

        return to_index, count+1

    def _count_bigram(self, to_index, vector_size, word_standardizer, page, cache):
        if page.url in cache:
            #print '---Get features from cache for: ', page.url
            return cache[page.url]
        
        text = page.text
        features = [0]* (vector_size) #last one for oov

        words = text.split()
        words.append('<end>')

        w1 = '<begin>'
        for word in words:
            w2 = word
            if word_standardizer is not None:
                w2 = word_standardizer(w2)

            index = vector_size-1
            if w1 in to_index and w2 in to_index[w1]:
                index = to_index[w1][w2] 
            
            features[index] +=1
            w1 = w2

        cache[page.url] = features
        self.cache_changed = True
        return features

    def get_features(self, en_page, fr_page):
        en_vec = self._count_bigram(self.en_bigram_to_index, self.en_vector_size, self.en_word_standardizer, en_page, self.en_cache_features)
        fr_vec = self._count_bigram(self.fr_bigram_to_index, self.fr_vector_size, self.fr_word_standardizer, fr_page, self.fr_cache_features)
        extra_vec = self._get_extra_features(en_page, fr_page)

        ret_vec = []
        ret_vec.extend(en_vec)
        ret_vec.extend(fr_vec)
        ret_vec.extend(extra_vec)

        return ret_vec

#================================For testing========================
#from ml_app.ngram_builder.bigram_builder import BigramBuilder

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    #has chanbed code, it couldn't run now, how to chnage the construction call
    bigram_builder = BigramBuilder(config)
    occur_above = 5
    en_bigram, count= bigram_builder.get_top_bigram('en', top_n=occur_above)
    print '---Total en_bigram used: ', count
    fr_bigram, count= bigram_builder.get_top_bigram('fr', top_n=occur_above)
    print '---Total fr_bigram used: ', count

    en_page = Page('url_en', 'html', 'and the in the', 'min_type', 'encode', 'lang')
    fr_page = Page('url_fr', 'html', 'text fr', 'min_type', 'encode', 'lang')

    fbuilder = BigramCounter(config, en_bigram, fr_bigram)
    fs = fbuilder.get_features(en_page, fr_page)
    print 'features 1: ', fs
    fs = fbuilder.get_features(en_page, fr_page)
    print 'features 2: ', fs
    en_page = Page('url_new', 'html', 'and the in the', 'min_type', 'encode', 'lang')
    fs = fbuilder.get_features(en_page, fr_page)
    print 'features 3: ', fs

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

