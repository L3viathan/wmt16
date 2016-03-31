import autopath
from ml_app.classifier.classifier import Classifier
from ml_app.data_provider.full_data_provider import FullDataProvider

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    corpus = FullDataProvider(config)
    classifer = Classifier(config, corpus)
    classifer.train()

def main(_):
    get_config()
    test1()

if __name__ == '__main__':
    main()

