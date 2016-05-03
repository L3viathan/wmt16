import sys

def print_a_predict(en_url, tops):
    print '%s\t%s'%(en_url, tops[0])
    #print '%s\t%s'%(en_url, tops[1])

with open('thanh.final.output.tsv', 'rt') as f:
    tops = []
    en_url = None
    for line in f:
        line = line.strip()
        if line.startswith('en_source_url'):
            if en_url is not None:
                print_a_predict(en_url, tops)
            en_url = line[line.find('http'):]
            tops = []
        elif len(line.split('\t'))==3:
            idx, score, fr_url = line.split('\t')
            tops.append(fr_url)
        else:
            sys.stderr.write('igenored:' + line + '\n')
