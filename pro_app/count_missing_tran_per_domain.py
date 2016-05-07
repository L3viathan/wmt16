import pdb
from app_funs import get_domain

def read_output_thanh():
    fname = './train_debugs/debug_full_05.txt'
    en_url = ''
    can_url = ''
    found_can = False
    cal_pairs = {}
    old_domain = ''
    count = 0
    with open(fname, 'rt') as f:
        for line in f:
            line = line.strip()
            ms = line.split()
            if len(ms)!=2:
                continue
            key, val = ms
            
            if key.startswith('----------'):
                en_url = line.split(' ')[1]
                found_can = False
                domain = get_domain(en_url)
                if old_domain is not None and domain != old_domain:
                    print old_domain , '\t', count
                    old_domain = domain
                    count = 0
            elif found_can==False and key.find(':')>=0:
                found_can = True
                can_url = val
                #print en_url + '\t' + can_url
                cal_pairs[en_url] = can_url
            elif key.startswith('---gold'):
                rank = int(key.replace('=', ',').split(',')[1])
                if rank==-1:
                    count +=1
            
    return cal_pairs


if __name__ == '__main__':
    read_output_thanh()

