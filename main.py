import sys
import gzip
import glob
import heapq
import itertools
from operator import itemgetter
from collections import namedtuple, defaultdict
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

def recall_at_n(n, assignments, gold):
    return sum(
            1
            for key in gold
            if gold[key] in map(itemgetter(1), heapq.nlargest(n, assignments[key]))
            ) / len(gold)


if __name__ == '__main__':
    # evaluation data
    gold = read_gold("data/train.pairs")

    # get input data, file by file.
    for filename in glob.iglob("data/*.lott.gz"):
        log("Now working on {}".format(filename))
        pairs = itertools.combinations(get_pages(filename), 2)

        for fltr in filters:
            # set up filters
            pairs = fltr(pairs)

        pairs = list(pairs)  # run all filters

        for scorer in scorers:
            # set up scorers
            top_n_size = 10
            assignments = defaultdict(list)
            for pair, score in scorer(pairs):
                left, right = pair
                if len(assignments[left]) < top_n_size:
                    heapq.heappush(assignments[left], (score, right))
                else:
                    heapq.heappushpop(assignments[left], (score, right))

            # now we have a dictionary from source document to a heap of (up to)
            # 10 target documents (with score).


            print("Recall@1:", recall_at_n(1, assignments, gold))
            print("Recall@5:", recall_at_n(5, assignments, gold))
            print("Recall@10:", recall_at_n(10, assignments, gold))
