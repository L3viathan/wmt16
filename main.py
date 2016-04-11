import sys
import gzip
import glob
import threading
import itertools
import subprocess
from collections import namedtuple
from base64 import b64decode

import simple

filters = [simple.filter]
scorers = [simple.score]

Page = namedtuple("Page", "url, html, text, mime_type, encoding, lang")

def get_pages(filename):
    with gzip.open(filename, mode="rt", encoding="utf-8") as f:
        for line in f:
            lang, mime, enc, url, html, text = line[:-1].split("\t")

            html = b64decode(html)
            text = b64decode(text)

            yield Page(url, html, text, mime, enc, lang)


def read_gold(filename):
    """Return pairs (english, french) of URLs that are gold pairs."""
    gold = {}
    with open(filename) as f:
        for line in f:
            source, target = line.strip().split("\t")
            gold[source] = target
    return gold

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':
    # get input data, file by file.
    for filename in glob.iglob("data/*.lott.gz"):
        log("Now working on {}".format(filename))
        pairs = itertools.combinations(get_pages(filename), 2)

    for fltr in filters:
        # set up filters
        pairs = fltr(pairs)

    for scorer in scorers:
        # set up scorers
        pairs = scorer(pairs)

    pairs = list(pairs)

    """
    for pair, score in pairs:
        left, right = pair
        print(left.url, right.url, score)
    """

    threshold = 0.5  # later: learned

    # evaluate
    gold = read_gold("data/train.pairs")
    ours = {}

    for pair, score in pairs:
        left, right = pair
        if score > threshold and left not in ours:
            ours[left] = right

    recall = sum(1 for key in gold if gold[key] == ours.get(key, None)) / len(gold)
    print("Recall:", recall)
