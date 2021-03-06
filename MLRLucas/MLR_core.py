# coding=utf-8
'''
使用 TensorFlow 建立 MLR 网络
'''

import tensorflow as tf

class MLRCore(object):
    def __init__(self, m=2, seed=None, data_length=None,
                 batch_size=-1, n_features=None, l2=0,
                 learning_rate_base=0.88, learning_rate_decay=0.999):
        '''

        :param m:
        :param seed:
        :param data_length:
        :param batch_size:
        :param n_features:
        :param l2:
        :param learning_rate_base: 初始学习率，如果训练轮数较少，可以设置得小一些
        :param learning_rate_decay: 学习率衰减率，与learning_rate_base联合使用，控制学习率衰减速度
            learning_rate_base、learning_rate_decay共同控制学习率，如果训练轮次很多，则应该让learning_rate_base比较大，
            让算法保存一个较长时间的大学习率模式
        '''
        self.m = m
        self.seed = seed
        self.data_length = data_length
        self.n_features = n_features
        self.graph = None
        self.learning_rate_base = learning_rate_base
        self.moving_average_decay = 0.999
        self.learning_rate_decay = learning_rate_decay
        self.l2_regularization_rate = l2
        self.batch_size = batch_size

    def set_num_features(self, n):
        self.n_features = n

    def init_placeholders(self):
        self.X_train = tf.placeholder(tf.float32, shape=[None, self.n_features], name='input_x')
        self.y_train = tf.placeholder(tf.float32, shape=[None], name='input_y')
        self.w_train = tf.placeholder(tf.float32, shape=[None], name='sample_weights')

    def init_main_block(self, regularizer):
        with tf.variable_scope('weights'):
            u = tf.get_variable('u', [self.n_features, self.m], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.5))
            w = tf.get_variable('w', [self.n_features, self.m], tf.float32, initializer=tf.random_normal_initializer(0.0, 0.5))
        # 正则化权重
        if regularizer:
            tf.add_to_collection('losses', regularizer(u))
            tf.add_to_collection('losses', regularizer(w))

        U = tf.matmul(self.X_train, u)
        p1 = tf.nn.softmax(U)
        W = tf.matmul(self.X_train, w)
        p2 = tf.nn.sigmoid(W)
        self.outputs = tf.reduce_sum(tf.multiply(p1, p2), 1)

    def init_loss(self):
        cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.outputs, labels=self.y_train) * self.w_train)
        self.loss = tf.add_n([cross_entropy_mean]) + tf.add_n(tf.get_collection('losses')) if self.l2_regularization_rate else tf.add_n([cross_entropy_mean])
        tf.summary.scalar('loss', self.loss)

    def init_target(self):
        with tf.name_scope('target'):
            self.checked_target = tf.verify_tensor_all_finite(self.loss,
                                                              msg='NaN or Inf in target value',
                                                              name='target')
            tf.summary.scalar('target', self.checked_target)

    def init_learning_rate(self):
        self.learning_rate = tf.train.exponential_decay(self.learning_rate_base,
                                                        self.global_steps,
                                                        self.data_length / self.batch_size,
                                                        self.learning_rate_decay)
        tf.summary.scalar('learning_rate', self.learning_rate)


    def init_moving_average(self):
        self.global_steps = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, self.global_steps)
        self.variable_averages_op = variable_averages.apply(tf.trainable_variables())
        self.variable_to_restore = variable_averages.variables_to_restore()
        tf.summary.scalar('global_steps', self.global_steps)

    def build_graph(self, restore=None):
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('input_block'):
                self.init_placeholders()
            with tf.name_scope('main_block'):
                # 弄个默认的正则化函数
                if self.l2_regularization_rate:
                    regularizer = tf.contrib.layers.l2_regularizer(float(self.l2_regularization_rate))
                else:
                    regularizer = None
                self.init_main_block(regularizer)
            with tf.name_scope('optimization_criterion'):
                self.init_moving_average()
                self.init_learning_rate()
                self.init_loss()
                self.init_target()
            self.trainer_step = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss, self.global_steps)
            with tf.control_dependencies([self.trainer_step, self.variable_averages_op]):
                self.trainer_op = tf.no_op(name='train')
            self.init_all_vars = tf.global_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            if restore:
                self.saver = tf.train.Saver(self.variable_to_restore)
            else:
                self.saver = tf.train.Saver()

