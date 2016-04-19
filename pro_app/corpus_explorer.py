from __future__ import print_function

import os
import gzip

data_path = '../ml_app/data'#Path to the corpus
train_pairs = os.path.join(data_path, 'train.pairs')
lett_path = os.path.join(data_path, 'lett.train')
tran_en = os.path.join(data_path, 'translations.train/url2text.en')


translation_urls = []

def get_traslated_urls():
    global tranlsated_urls
    print('Get translated urls....')
    with open(tran_en, 'rt') as f:
        url = ''
        for line in f:
            url_new, sen = line.split('\t')
            if url != url_new:
                url = url_new
                translation_urls.append(url)

def miss_translation():
    for d_file in os.listdir(lett_path):
        print('-'*20, d_file)
        #if d_file != 'cineuropa.mobi.lett.gz':
            #continue
        d_file = os.path.join(lett_path, d_file)
        f = gzip.GzipFile(d_file, mode='r')
        for line in f:
            url = line.split('\t')[3]
            if url not in translation_urls:
                print(url)

def main():
    get_traslated_urls()
    miss_translation()

if __name__ == '__main__':
    main()

