from collections import Counter
scores = {}

with open("missing_translations.txt") as f:
    for line in f:
        line = line.lstrip("-").split()
        if line:
            scores[line[0][:-8]] = eval(line[-1])

with open("than_top10_training.txt") as than:
    print("method,missingpct,domain,recallat1")
    next(than)
    for line in than:
        if line.startswith("---domain"):
            fqdn = line.split()[1]
            print("than",scores[fqdn],fqdn,line.split("top1: ")[1].split(",")[0], sep=",")
        else:
            break

with open("train.pairs") as f:
    gold = {}
    for line in f:
        source, target = line.split()
        gold[source] = target

def fqdn(url):
    return url.split("/")[2]  # very dirty, but works

hoa_results = """iiz-dvv.de 0.985074626866
creationwiki.org    0.909090909091
minelinks.com   0.969696969697
www.prohelvetia.ch  0.714285714286
www.dfo-mpo.gc.ca   1.0
www.usw.ca  0.78313253012
manchesterproducts.com  0.5
www.pawpeds.com 0.976744186047
www.eu2007.de   0.272727272727
www.acted.org   0.857142857143
www.summerlea.ca    0.965517241379
www.bugadacargnel.com   0.571428571429
eu.blizzard.com 1.0
tsb.gc.ca   0.983050847458
ironmaidencommentary.com    0.390243902439
www.nauticnews.com  0.761904761905
www.lameca.org  1.0
cineuropa.mobi  0.958904109589
santabarbara-online.com 1.0
www.eohu.ca 1.0
www.the-great-adventure.fr  0.833333333333
galacticchannelings.com 1.0
www.ushmm.org   0.961538461538
bugadacargnel.com   0.631578947368
virtualhospice.ca   0.760869565217
golftrotter.com 1.0
www.socialwatch.org 0.238095238095
www.krn.org 0.970149253731
www.artsvivants.ca  0.75
kustu.com   1.0
www.cyberspaceministry.org  1.0
www.nato.int    0.583333333333
cbsc.ca 1.0
kicktionary.de  1.0
www.antennas.biz    1.0
www.vinci.com   1.0
www.eu2005.lu   0.882352941176
eu2007.de   0.454545454545
rehazenter.lu   0.6875
schackportalen.nu   1.0
www.dakar.com   0.888888888889
www.inst.at 1.0
pawpeds.com 0.947368421053
www.fao.org 1.0
www.luontoportti.com    0.633333333333
www.bonnke.net  1.0
www.ec.gc.ca    0.961538461538
forcesavenir.qc.ca  0.5
www.cgfmanet.org    1.0""".split("\n")

for line in hoa_results:
    domain, results = line.split()
    print("hoa",scores[domain], domain, results, sep=",")
