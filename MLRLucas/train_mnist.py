# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

class TrainModel(object):
    def __init__(self):
        self.INPUT_NODE = 784
        self.OUTPUT_NODE = 10
        self.LAYER1_NODE = 500

        self.BATCH_SIZE = 100
        self.LEARNING_RATE_BASE = 0.8
        self.LEARNING_RATE_DECAY = 0.99
        self.REGULARIZATION_RATE = 0.0001
        self.TRAINING_STEPS = 30000
        self.MOVING_AVERAGE_DECAY = 0.99
        self.MODEL_SAVE_PATH = './model/'
        self.MODEL_NAME = 'model.ckpt'

        self.graph = None
        self.saver = None
        self.FEATURE_LENGTH = 0
        self.DATA_LENGTH = 0

    def set_data_shape(self, shape):
        self.DATA_LENGTH = shape[0]
        self.FEATURE_LENGTH = shape[1]

    def get_weight_variable(self, shape, regularizer):
        weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer:
            tf.add_to_collection('losses', regularizer(weights))
        return weights

    def inference(self, input_tensor, regularizer, reuse=None):
        with tf.variable_scope('layer1', reuse=reuse) as scope:
            weights = self.get_weight_variable([self.INPUT_NODE, self.LAYER1_NODE], regularizer)
            baises = tf.get_variable('biases', [self.LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + baises)
            scope.reuse_variables()
        with tf.variable_scope('layer2', reuse=reuse) as scope:
            weights = self.get_weight_variable([self.LAYER1_NODE, self.OUTPUT_NODE], regularizer)
            baises = tf.get_variable('biases', [self.OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + baises
            scope.reuse_variables()
        return layer2

    def train(self, mnist, reuse=None):
        x = tf.placeholder(tf.float32, [None, self.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, self.OUTPUT_NODE], name='y-input')
        regularizer = tf.contrib.layers.l2_regularizer(self.REGULARIZATION_RATE)
        y = self.inference(x, regularizer, reuse=reuse)
        globals_step = tf.Variable(0, trainable=False)

        # with tf.variable_scope('layer1', reuse=reuse) as scope:
        variable_averages = tf.train.ExponentialMovingAverage(self.MOVING_AVERAGE_DECAY, globals_step)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())
        # scope.reuse_variables()

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(self.LEARNING_RATE_BASE,
                                                   globals_step,
                                                   self.DATA_LENGTH / self.BATCH_SIZE,
                                                   self.LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=globals_step)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

        self.saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(self.BATCH_SIZE)
                _, loss_value, step = sess.run([train_op, loss, globals_step], feed_dict={x: xs, y_: ys})
                if i % 100 == 0:
                    print('After %d training step(s), loss on training batch is  %g.' % (step, loss_value))
                    self.saver.save(sess, os.path.join(self.MODEL_SAVE_PATH, self.MODEL_NAME), global_step=globals_step)

def main(argv=None):
    mnist_data_path = 'data'
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)
    trainModel = TrainModel()
    trainModel.set_data_shape(mnist.train.images.shape)
    if trainModel.graph:
        print('trainModel graph is not None')
        trainModel.train(mnist, reuse=True)
    else:
        print('trainModel graph is None')
        trainModel.train(mnist, reuse=False)

if __name__ == '__main__':
    tf.app.run()

