# coding=utf-8
'''
使用 TensorFlow 建立 MLR 网络
'''

import tensorflow as tf

class MLRCore(object):
    def __init__(self, m=2, seed=None, data_length=None,
                 batch_size=-1, n_features=None):
        self.m = m
        self.seed = seed
        self.data_length = data_length
        self.n_features = n_features
        self.graph = None
        self.learning_rate_base = 0.88
        self.moving_average_decay = 0.999
        self.learning_rate_decay = 0.999
        self.batch_size = batch_size

    def set_num_features(self, n):
        self.n_features = n

    def init_placeholders(self):
        self.X_train = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_x')
        self.y_train = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.w_train = tf.placeholder(tf.float32, shape=[None], name='sample_weights')

    def init_main_block(self):
        with tf.variable_scope('weights'):
            u = tf.get_variable('u', [self.n_features, self.m], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.5))
            w = tf.get_variable('w', [self.n_features, self.m], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.5))

        U = tf.matmul(self.X_train, u)
        p1 = tf.nn.softmax(U)
        W = tf.matmul(self.X_train, w)
        p2 = tf.nn.sigmoid(W)
        self.outputs = tf.reduce_sum(tf.multiply(p1, p2), 1)

    def init_loss(self):
        cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.y_train) * self.w_train)
        self.loss = tf.add_n([cross_entropy_mean])

    def init_target(self):
        with tf.name_scope('target'):
            self.checked_target = tf.verify_tensor_all_finite(self.loss,
                                                              msg='NaN or Inf in target value',
                                                              name='target')

    def init_learning_rate(self):
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base,
                                                        self.global_steps,
                                                        self.data_length / self.batch_size,
                                                        self.learning_rate_decay)

    def init_moving_average(self):
        self.global_steps = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_steps)
        self.variable_averages_op = variable_averages.apply(tf.trainable_variables())
        self.variable_to_restore = variable_averages.variables_to_restore()

    def build_graph(self, restore=None):
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('input_block'):
                self.init_placeholders()
            with tf.name_scope('main_block'):
                self.init_main_block()
            with tf.name_scope('optimization_criterion'):
                self.init_moving_average()
                self.init_learning_rate()
                self.init_loss()
                self.init_target()
            self.trainer_step = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss, self.global_steps)
            with tf.control_dependencies([self.trainer_step, self.variable_averages_op]):
                self.trainer_op = tf.no_op(name='train')
            self.init_all_vars = tf.global_variables_initializer()
            if restore:
                self.saver = tf.train.Saver(self.variable_to_restore)
            else:
                self.saver = tf.train.Saver()
