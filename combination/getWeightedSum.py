with open("results_split_at_50.csv") as f:
    next(f)
    s = 0
    for line in f:
        domain, score, weight = line.split(",")
        s += float(score)*float(weight)
print(s)
