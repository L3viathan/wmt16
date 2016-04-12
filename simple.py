def filter(pairs):
    for pair in pairs:
        yield pair # demo: yield all pairs
        #break

def score(pairs):
    for pair in pairs:
        yield (pair, 0.834) # demo: fixed score
        #break

def filter_ranker_scorer(en_page, candidates, scores):
    '''An operation in pipline.
        
        params:
            en_page: the page object of the souce page we wish to find its transation
            candidates: dict of page object which are current best candidate for the source docuement en_page
            scores: numpy.ndarrray, scores corespoinding to candidates, so you might use the computation of previous operator

        returns:
            candidates: most potential candiates
            scores: socres of candiates (NOTE: don't remove other previous scores)

        Note for building operator:
            fileter operator: remove both candidats and repsective row in the scores
            score operator: add one column at the last and set the new score there
            ranker operator: rank the current candiate by the score provided

        Note for dividing operator:
            divide as small as possible, so other could user your operator again
    '''
    pass
