import sys

with open('thanh.output.txt', 'rt') as f:
    urls_covered = set()
    covered = True
    for line in f:
        line = line.strip()
        if line.startswith('------en_url_source:'):
            en_url = line[line.find('http'):]
            if en_url in urls_covered:
                covered = True
                sys.stderr.write('repeated url, igenored:' + line + '\n')
            else:
                covered = False
                urls_covered.add(en_url)
                print '--newpredict--'
                print 'en_source_url\t%s'%en_url
        elif not covered and len(line.split('\t'))==3:
            print line
        else:
            sys.stderr.write('igenored:' + line + '\n')
