from filters.length import LengthFilter
from filters.topn import TopNFilter
from scorers.sharedword import SimpleScorer
from rankers.simple import SimpleRanker
from simple import filter1, filter2, score1, score2, ranker1, ranker2

config = {
    'general':{
        'data_path': './data/lett.train',
        'gold_file': '/data/train.pairs',
        'source_urls': './data/souce_urls.txt',
    },
    'pipeline':[
        filter1,#filter to process at small number of candidate
        score1,#score candiate
        ranker1,#order the canidate by the score of socre1
        filter2,#since score2 very computational expensive, so get only top1000 best candiates which is sorted by ranker1
        score2,#really good score, but slow
        ranker2,#combine and rank the list of candidate again 
    ]
}
