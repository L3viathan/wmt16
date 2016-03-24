import autopath

class BigramCounter(object):
    
    def __init__(self, config):
        pass

    def get_feature(self, en_text, fr_text):
        pass


#================================For testing========================
config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    fextractor = BigramCounter(config)

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()

