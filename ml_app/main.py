import autopath
from ml_app.classifier.classifier import Classifier
from ml_app.data_provider.full_data_provider import FullDataProvider
import sys

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    classifier = Classifier(config)

    #NOTE: manualy switch/change the lines following
    #classifier.train()

    #'''
    classifier.load_best_model()#The current version classifier requires manuaaly set the trraning step which you think it is the best
   
    sys.stderr.write('start eval valid\n')
    print '--------------------------------start eval valid'
    top1, top5, top10 = classifier.evaluate_wmt16('./data/thanh/valid.txt', True)
    print 'Evaluation valid: top1: %.3f, top5: %.3f, top10: %.3f'%(top1, top5, top10)
    sys.stderr.write('Evaluation valid: top1: %.3f, top5: %.3f, top10: %.3f\n'%(top1, top5, top10))

    sys.stderr.write('start eval test\n')
    print '---------------------------------start eval test'
    top1, top5, top10 = classifier.evaluate_wmt16('./data/thanh/test.txt', True)
    print 'Evaluation valid: top1: %.3f, top5: %.3f, top10: %.3f'%(top1, top5, top10)
    sys.stderr.write('Evaluation test: top1: %.3f, top5: %.3f, top10: %.3f\n'%(top1, top5, top10))

    sys.stderr.write('start eval train\n')
    print '--------------------start eval train'
    top1, top5, top10 = classifier.evaluate_wmt16('./data/thanh/train.txt')
    print 'Evaluation train: top1: %.3f, top5: %.3f, top10: %.3f'%(top1, top5, top10)
    sys.stderr.write('Evaluation train: top1: %.3f, top5: %.3f, top10: %.3f\n'%(top1, top5, top10))
    #'''

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()
