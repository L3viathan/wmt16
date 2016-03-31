class DataSet(object):
    def __init__(self):
        pass

    def next_batch(self, batch_size):
        pass

class DataManager(object):
    def __init__(self):
        #readd all train, valid, dataset test into memory
        self.train = DataSet()
        self.valid = DataSet()
        self.test = DataSet()
