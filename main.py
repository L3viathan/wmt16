import sys
import gzip
import glob
import threading
import itertools
import subprocess

def read_gold(filename):
    """Return pairs (english, french) of URLs that are gold pairs."""
    gold = set()
    with open(filename) as f:
        for line in f:
            gold.add(tuple(line.strip().split("\t")))
    return gold

def call_filter(filter_name, pairs):
    """Given the name of an executable and an iterable of pairs,
    yield those pairs that are not filtered out."""
    p = subprocess.Popen(
            filter_name.split(" "),
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=8192,
            )

    def writer():
        for pair in pairs:
            p.stdin.write("\t".join(pair) + "\n")
        log("writer is done")
        p.stdin.close()  # send EOF

    thread = threading.Thread(target=writer)
    thread.start()

    yield from p.stdout
    thread.join()
    p.wait()

def log(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


if __name__ == '__main__':
    # read config
    from config import config

    # get input data, file by file.
    for filename in glob.iglob("data/*.lott.gz"):
        log("Now working on {}".format(filename))
        with gzip.open(filename, 'rt', encoding="utf-8") as f:
            lines = [line.split()[0] for line in f] # DEBUG # TODO remove the split shit
        print(lines[:10])
        pairs = itertools.combinations(lines, 2)

    # call filters
    for filter in config["filters"]:
        log("Building {}".format(filter))
        # build pipeline
        pairs = call_filter(filter, pairs)

    log(pairs)

    log("Executing pipeline")
    for thing in pairs:
        print(thing)
        ...

    # call scorers
    # evaluate
    gold = read_gold("data/train.pairs")
