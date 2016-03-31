from __future__ import division
#import numpy
#from six.moves import xrange 

import time
import tensorflow as tf

class Classifier(object):
    
    def __init__(self, config, datasets):
        self.full_config = config
        self.config = config['classifier']
        self.corpus = datasets
        self.model = self.config['model'](self.full_config)

        #these parametes will go to config for xeveral component
        self.learning_rate = 0.01
        self.max_step = 2000
        self.batch_size = 100

    def fill_feed_dict(self, dataset, X_pl, Y_pl):
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
                feed = self.fill_feed_dict(self.corpus.train, X_placeholder, Y_placeholder)
                
                _, loss_value = sess.run([train_op, loss_op], feed_dict=feed)
                duration = time.time() - start_time

                if step%100 == 0:
                    print 'Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration)
                    #summary_str = sess.run(summary_op, feed_dict=feed)
                    #summary_writer.add_summary(summary_str, step)

                if (step+1)%1000 == 0 or (step+1)==self.max_step:
                    saver.save(sess, './', global_step=step)
                    print 'Evaluate train set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.train)
                    print 'Evaluate valid set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.validation)
                    print 'Evaluate test set:'
                    self.evaluate(sess, eval_op, X_placeholder, Y_placeholder, self.corpus.test)

    def predict(self, X):
        pass

    def evaluate(self, sess, eval_op, X_pl, Y_pl, dataset):
        true_count = 0.0
        step_num = dataset.num_examples // self.batch_size
        num_examples = step_num*self.batch_size
        for step in xrange(step_num):
            feed = self.fill_feed_dict(dataset, X_pl, Y_pl)
            true_count += sess.run(eval_op, feed_dict=feed)
        precision = true_count / num_examples
        print '  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %(num_examples, true_count, precision)

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

def main(_):
    get_config()
    test1()

if __name__ == '__main__':
    main()
