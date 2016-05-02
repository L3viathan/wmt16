missclues = {}
with open("train.tran.miss.clues.tsv") as f:
    for line in f:
        *cols, ensourceurl = line.strip().split()
        missclues[ensourceurl] = cols

with open("out.tsv", "w") as fw, open("than_top10_training.txt") as f:
    print("source_url\ttop1prediction\tcorrect\ttranslation_missed\ttop1_score\tmin_score\ten_src_len\ten_source_vocab_len\tcol_len\tcol_vocab_len")
    for line in f:
        if not line.startswith("----------"): continue
        _, source = line.strip().split()
        line = next(f)
        correct = "rank=1" in line
        translation_missed = "maybe?"
        line = next(f)
        score, url = line.strip().split(":\t")
        print(source, url, correct, translation_missed, score, *missclues[source], sep="\t")

