import autopath

class BigramCounter(object):
    
    def __init__(self, config, en_bigram, fr_bigram):
        self.full_config = config
        self.config = config['feature_builder']['BigramCounter']
        self.en_word_standardizer = self.config['en_word_standardizer']
        self.fr_word_standardizer = self.config['fr_word_standardizer']
        self._get_extra_features = self.config['add_extra_features']

        self.en_bigram = en_bigram
        self.fr_bigram = fr_bigram
        self.en_bigram_to_index, en_size = self._format_vector(self.en_bigram)
        self.fr_bigram_to_index, fr_size = self._format_vector(self.fr_bigram)
        self.bigram_vector_size = en_sizr + fr_size + 1 #plust one for oov

        self.source_lang = 'en'
        self.target_lang = 'fr'

    def _format_vector(self, bigram):
        to_index = {}
        count = 0
        for w1 in bigram:
            to_index[w1] = {}
            for w2 in bigram[w2]:
                to_index[w1][w2] = count
                count +=1

        return to_index, count+1

    def _count_bigram(self, bigram, to_index, word_standardizer, text):
        feature = [0]*self.bigram_vector_size

        words = text.split()
        words.append('<end>')

        w1 = '<begin>'
        for word in words:
            w2 = word
            if word_standardizer is not None:
                w2 = word_standardizer(w2)

            index = self.bigram_vector_size + 1
            if w1 in to_index and w2 in to_index[w1]:
                index = to_index[w1][w2] 
            
            feature[index] +=1
            w1 = w2

        return feature

    def get_feature(self, en_page, fr_page):
        en_text = en_page.text
        fr_text = fr_page.text 

        en_vec = self._count_bigram(self.en_bigram, en_text)
        fr_vec = self._count_bigram(self.fr_bigram, fr_text)
        extra_vec = self._get_extra_features(en_page, fr_page)

        en_vec.extend(fr_vec).extend(extra_vec)

        return en_vec

#================================For testing========================
config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    fbuilder = BigramCounter(config)
    #TODO: next, test extract feature for a stence

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

