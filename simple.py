def filter(pairs):
    for pair in pairs:
        yield pair # demo: yield only the first pair
        #break

def score(pairs):
    for pair in pairs:
        yield (pair, 0.834) # demo: fixed score
        #break
