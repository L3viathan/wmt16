from __future__ import division

import math
import tensorflow as tf

class NeuralNetwork(object):
    def __init__(self, config):
        self.full_config = config
        self.config = config['learning_model']['NeuralNetwork']

        self.layer_description = self.config['layer_description']
        self.hidden_layers_desc = self.layer_description[1:]
        
        self.input_size = self.layer_description[0]['unit_size']#input size
        self.class_number = self.layer_description[-1]['unit_size']#output_size

    def inference(self, input_holder):
        last_layer = input_holder
        last_size = self.input_size 
        print '---Build NN, layer: input', last_size

        for layer in self.hidden_layers_desc:
            layer_name = layer['name']
            layer_size = layer['unit_size']
            print '---Build NN, layer:', layer_name, layer_size
            with tf.name_scope(layer_name):
                weights = tf.Variable(tf.truncated_normal([last_size, layer_size], stddev=1.0/math.sqrt(float(last_size))), name='weights')
                biases = tf.Variable(tf.zeros([layer_size], name='biases'))
                if layer['active_fun'] is None:
                    hidden_layer = tf.matmul(last_layer, weights) + biases
                else:
                    hidden_layer = layer['active_fun'](tf.matmul(last_layer, weights) + biases)
                last_layer = hidden_layer
                last_size = layer_size

        return last_layer

    def loss(self, out_logits_layer, labels):
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size), 1)
        concated = tf.concat(1, [indices, labels])
        onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, self.class_number]), 1.0, 0.0)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(out_logits_layer, onehot_labels, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        return loss

    def training(self, loss, learning_rate):
        tf.scalar_summary(loss.op.name, loss)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return train_op

    def evaluation(self, out_logits_layer, labels):
        correct = tf.nn.in_top_k(out_logits_layer, labels, 1)
        
        return tf.reduce_sum(tf.cast(correct, tf.int32))
