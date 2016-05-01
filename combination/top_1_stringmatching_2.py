import sys
from Levenshtein import distance
from collections import Counter
scores = {}

threshold = 0.35

def fqdn(url):
    return url.split("/")[2]  # very dirty, but works

with open("missing_translations.txt") as f:
    for line in f:
        line = line.lstrip("-").split()
        if line:
            scores[line[0][:-8]] = eval(line[-1])

with open("than_top10_training.txt") as than:
    for line in than:
        if line.startswith("----------"):
            _, source = line.strip().split()
            next(than)  # ignore gold
            domain = fqdn(source)
            if scores[domain] < threshold:
                continue
            score1, match1 = next(than).strip().split()
            score2, match2 = next(than).strip().split()
            if abs(float(score1[:-1])-float(score2[:-1])) < 70:
                print(source, min([match1, match2], key=lambda x: distance(x, source)), sep="\t")
            else:
                print(source, match1, sep="\t")
        else:
            continue

seen = set()
with open("hoa_top10_training.txt") as hoa:
    for line in hoa:
        try:
            source, match, score = line.strip().split()
        except ValueError:
            continue
        if source in seen:
            continue
        domain = fqdn(source)
        if scores[domain] < threshold:
            print(source, match, sep="\t")
            seen.add(source)
