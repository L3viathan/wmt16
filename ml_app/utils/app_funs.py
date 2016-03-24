import base64
import gzip
from collections import namedtuple

Page = namedtuple(
    "Page", "url, html, text, mime_type, encoding, lang")

def get_domain(url):
    return url[7: url.index('/', 7)]

def read_gold(filename):
    """Return pairs (english, french) of URLs that are gold pairs."""
    gold = set()
    with open(filename) as f:
        for line in f:
            gold.add(tuple(line.strip().split("\t")))
    return gold

def read_lett_iter(f, decode=True):
    fh = f
    if f.name.endswith('.gz'):
        fh = gzip.GzipFile(fileobj=fh, mode='r')
    for line in fh:
        lang, mine, enc, url, html, text = line[:-1].split("\t")

        html = base64.b64decode(html)
        text = base64.b64decode(text)

        if decode:
            html = html.decode("utf-8")
            text = text.decode("utf-8")

        p = Page(url, html, text, mine, enc, lang)
        yield p

def read_lett(f, source_language, target_language):
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
    return source_corpus, target_corpus


def get_extra_features(en_page, fr_page):
    return []
