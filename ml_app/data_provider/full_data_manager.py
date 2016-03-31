class DataSet(object):
    def __init__(self, X, Y):
        pass

    def next_batch(self, batch_size):
        pass

class FullDataManager(object):
    def __init__(self, config, feature_builder):
        #readd all train, valid, dataset test into memory
        self.full_config = config
        self.config = config['data_manager']['FullDataManager']
        self.feature_builder = feature_builder
    
        self.train_file = self.config['train_file']
        self.valid_file = self.config['valid_file']
        self.test_file = self.config['test_file']

        self.train = DataSet()
        #self.valid = DataSet()
        #self.test = DataSet()

    def buid_data_set(corpus_file):
        X = 0#
        Y =
        lines = []
        with open(corpus_file, 'rt') as f:
            lines = f.readlines()

        num = len(lines)
        vector_size = feature_builder.vector_size
        X = np.zeros(shape=(num, vector_size))

                

#================================For testing========================
config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    pass

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

