def read_gold(filename):
    """Return pairs (english, french) of URLs that are gold pairs."""
    gold = set()
    with open(filename) as f:
        for line in f:
            gold.add(tuple(line.strip().split("\t")))
    return gold

if __name__ == '__main__':
    gold = read_gold("data/train.pairs")
    print(gold)
