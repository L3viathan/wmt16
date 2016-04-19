from __future__ import division
#import numpy
#from six.moves import xrange 

import time
import numpy as np
import tensorflow as tf

from ml_app.utils.io_funs import text_file_line_iter

class Classifier(object):
    
    def __init__(self, config):
        self.full_config = config
        self.config = config['classifier']
        self.model = self.config['model'](self.full_config)
        self.corpus = self.config['data_provider'](self.full_config)
        #TODO: a better switch between train and evaluate, don't load all data when evaluate, since not used it

        self.learning_rate = self.config['learning_rate']
        self.max_step = self.config['max_step']
        self.batch_size = self.config['batch_size']
        self.step_to_report_loss = self.config['step_to_report_loss']
        self.step_to_save_eval_model = self.config['step_to_save_eval_model']
        self.model_storage_file = self.config['model_storage_file']
        self.load_model_step = self.config['load_model_step']

    def fill_feed_dict(self, dataset, X_pl, Y_pl, balanced=False):
        if balanced:
            X, Y = dataset.next_balanced_batch(self.batch_size)
        else:
            X, Y = dataset.next_batch(self.batch_size)
        return {X_pl: X, Y_pl: Y}

    def train(self):
        with tf.Graph().as_default():
            X_placeholder = tf.placeholder(tf.float32, shape=(None, None))
            Y_placeholder = tf.placeholder(tf.int32, shape=(None))

            predict_op = self.model.inference(X_placeholder)
            loss_op = self.model.loss(predict_op, Y_placeholder)
            train_op = self.model.training(loss_op, self.learning_rate)
            eval_op = self.model.evaluation(predict_op, Y_placeholder)

            #summary_op = tf.merge_all_summaries()

            saver = tf.train.Saver()
            sess = tf.Session()
            init = tf.initialize_all_variables()
            sess.run(init)

            #summary_writer = tf.train.SummaryWriter('./', graph_def=sess.graph_def)
            
            self.eval_op = eval_op
            self.sess = sess

            for step in xrange(self.max_step):
                start_time = time.time()
                feed = self.fill_feed_dict(self.corpus.train, X_placeholder, Y_placeholder, balanced=True)
                
                _, loss_value = sess.run([train_op, loss_op], feed_dict=feed)
                duration = time.time() - start_time

                if step%self.step_to_report_loss == 0:
                    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                    #summary_str = sess.run(summary_op, feed_dict=feed)
                    #summary_writer.add_summary(summary_str, step)

                if (step+1)%self.step_to_save_eval_model == 0 or (step+1)==self.max_step:
                    saver.save(sess, self.model_storage_file, global_step=(step+1))
                    print 'Evaluate train set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.train)
                    print 'Evaluate valid set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.validation)
                    print 'Evaluate test set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.test)

    def predict(self, X):
        return  self.sess.run(self.predict_op, feed_dict={self.X_placeholder:X})

    def evaluate(self, sess, eval_op, X_pl, Y_pl, dataset):
        true_count = 0.0
        step_num = dataset.num_examples // self.batch_size
        num_examples = step_num*self.batch_size
        for step in xrange(step_num):
            feed = self.fill_feed_dict(dataset, X_pl, Y_pl)
            true_count += sess.run(eval_op, feed_dict=feed)
        precision = true_count / num_examples
        print '  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision)

    def load_best_model(self):
         with tf.Graph().as_default():
            self.X_placeholder = tf.placeholder(tf.float32, shape=(None, None))
            self.Y_placeholder = tf.placeholder(tf.int32, shape=(None))

            self.predict_op = self.model.inference(self.X_placeholder)
            self.loss_op = self.model.loss(self.predict_op, self.Y_placeholder)
            self.train_op = self.model.training(self.loss_op, self.learning_rate)
            self.eval_op = self.model.evaluation(self.predict_op, self.Y_placeholder)

            self.saver = tf.train.Saver()
            self.sess = tf.Session()
            print '---Loading the model at training step ', self.load_model_step
            self.saver.restore(self.sess, self.model_storage_file + '-' + str(self.load_model_step))

    def evaluate_wmt16(self, gold_file, debug=False):
        #print 'Evaluate valid set:'
        #self.evaluate(self.sess, self.eval_op, self.X_placeholder, self.Y_placeholder, self.corpus.validation)
        test_count = top1_score = top5_score = top10_score = 0.0
        debug_print = ''
        for line in text_file_line_iter(gold_file):
            test_count += 1
            en_url, fr_url = line.strip().split('\t')
            en_url = en_url.strip()
            gold_url = fr_url.strip()

            X_corpus, fr_urls = self.corpus.get_one_url_corpus(en_url)
            gold_idx = fr_urls.index(gold_url)
            scores = self.predict(X_corpus)
            predict_labels = np.argmax(scores, 1)
            positive_idxs = np.where(predict_labels==1)[0]
            
            if len(positive_idxs)==0:
                print '!!!---WARNING no positive pairs found:', en_url
                top10_idxs = np.argsort(scores[:, 1])[-10:][::-1] #TODO check this one next time
            else:
                can_scores = scores[:, 1][positive_idxs]
                sorted_idxs = np.argsort(can_scores)[-10:][::-1]
                top10_idxs = positive_idxs[sorted_idxs]

            if gold_idx == top10_idxs[0]:
                debug_print = 'CORRECT'
                top1_score += 1
                top5_score += 1
                top10_score += 1
            elif gold_idx in top10_idxs[0:5]:
                top5_score += 1
                top10_score += 1
                debug_print = 'in TOP5'
            elif gold_idx in top10_idxs[0:10]:
                top10_score += 1
                debug_print = 'in TOP10'
            else:
                debug_print = 'WRONG out of top10'

            if debug:
                gold_idx_rank = np.where(np.argsort(scores[:, 1])[::-1]==gold_idx)[0][0]
                print '*****------%s'%(en_url)
                print '--rs:%s'%(debug_print)
                print '--can_num:%d'%(len(fr_urls))
                print '--gold(%d):%s'%(gold_idx_rank, gold_url)
                print '---top 10 cans:'
                print '\n'.join(np.array(fr_urls)[top10_idxs])

        top1_score = top1_score/test_count
        top5_score = top5_score/test_count
        top10_score = top10_score/test_count
        return top1_score, top5_score, top10_score

#================================For testing========================
from tensorflow.examples.tutorials.mnist import input_data

import autopath
from ml_app.models.nn import NeuralNetwork

config= None

def get_config():
    global config
    from ml_app.config import config as cf
    config = cf

def test1():
    corpus = input_data.read_data_sets('/scratch/home/thanh/study/machine_learning/libraries/tensorflow/mnist/data')
    classifer = Classifier(config, corpus)
    classifer.train() 

def main():
    get_config()
    test1()

if __name__ == '__main__':
    main()
