from __future__ import print_function

import os
import sys
import gzip

'''
data_path = '../ml_app/data'#Path to the corpus
lett_path = os.path.join(data_path, 'lett.train')
tran_en = os.path.join(data_path, 'translations.train/url2text.en')
'''

data_path = '../ml_app/data/test'#Path to the corpus
lett_path = os.path.join(data_path, 'lett.test')
tran_en = os.path.join(data_path, 'translations.test/url2text.en.detok')
#'''

translation_urls = []

def get_translated_urls():
    #print('Get translated urls....')
    sys.stderr.write('Get translated urls...\n')
    with open(tran_en, 'rt') as f:
        url = ''
        for line in f:
            url_new, sen = line.split('\t')
            if url != url_new:
                url = url_new
                translation_urls.append(url)

def miss_translation():
    g_total_count, g_miss_count = 0, 0
    for d_file in os.listdir(lett_path):
        sys.stderr.write('-'*10 +  d_file)
        total_count = 0
        miss_count = 0
        #if d_file != 'cineuropa.mobi.lett.gz':
            #continue
        d_file = os.path.join(lett_path, d_file)
        f = gzip.GzipFile(d_file, mode='r')
        for line in f:
            total_count += 1
            g_total_count += 1
            url = line.split('\t')[3]
            if url not in translation_urls:
                print(url)
                miss_count += 1
                g_miss_count += 1

        sys.stderr.write(' Missed: %d/%d\n'%(miss_count, total_count))
    sys.stderr.write(' Global Missed: %d/%d\n'%(g_miss_count, g_total_count))

if __name__ == '__main__':
    get_translated_urls()
    miss_translation()
