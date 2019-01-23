# encoding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import time, os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def get_data():
    train_data = pd.read_csv("data\\adult.data.txt", header=None, delimiter=',', encoding='GB2312')
    test_data = pd.read_csv("data\\adult.test.txt", header=None, delimiter=',', encoding='GB2312')

    all_columns = ['age','workclass','fnlwgt','education','education-num',
                        'marital-status','occupation','relationship','race','sex',
                        'capital-gain','capital-loss','hours-per-week','native-country','label','type']

    continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    dummy_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']

    train_data['type'] = 1
    test_data['type'] = 2

    all_data = pd.concat([train_data,test_data],axis=0)
    all_data.columns = all_columns

    all_data = pd.get_dummies(all_data,columns=dummy_columns)
    test_data = all_data[all_data['type']==2].drop(['type'],axis=1)
    train_data = all_data[all_data['type']==1].drop(['type'],axis=1)

    train_data['label'] = train_data['label'].map(lambda x: 1 if x.strip() == '>50K' else 0)
    test_data['label'] = test_data['label'].map(lambda x: 1 if x.strip() == '>50K.' else 0)

    for col in continus_columns:
        ss = StandardScaler()
        train_data[col] = ss.fit_transform(train_data[[col]])
        test_data[col] = ss.transform(test_data[[col]])

    train_y = train_data['label']
    train_x = train_data.drop(['label'],axis=1)
    test_y = test_data['label']
    test_x = test_data.drop(['label'],axis=1)

    return train_x, train_y, test_x, test_y

class MLRClass(object):
    def __init__(self):
        self.feature_length = 10
        self.data_length = 20
        self.moving_average_decay = 0.999
        self.learning_rate_base = 0.8
        self.learning_rate_decay = 0.999
        self.batch_size = 100
        self.model_save_path = 'model'
        self.model_save_name = 'model.ckpt'
        self.graph = None
        self.m = 2
        self.n_epochs = 100

    def set_data_shape(self, shape):
        self.data_length = shape[0]
        self.feature_length = shape[1]

    def set_fragmentations(self, m):
        self.m = m

    def set_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_batch_size(self, b):
        self.batch_size = b

    def inference(self, m, input_tensor, X_dim):
        with tf.variable_scope('weights') as scp:
            u = tf.get_variable(name='u', shape=[X_dim, m], initializer=tf.random_normal_initializer(0.0, 0.5))
            w = tf.get_variable(name='w', shape=[X_dim, m], initializer=tf.random_normal_initializer(0.0, 0.5))
            scp.reuse_variables()

        U = tf.matmul(input_tensor, u)
        p1 = tf.nn.softmax(U)
        W = tf.matmul(input_tensor, w)
        p2 = tf.nn.sigmoid(W)
        return tf.reduce_sum(tf.multiply(p1, p2), 1)

    def train(self, m=2, learning_rate=0.51, n_epoch=1000):
        train_x, train_y, test_x, test_y = get_data()
        self.set_data_shape(train_x.shape)
        X_dim = train_x.shape[1]
        X = tf.placeholder(tf.float32, [None, X_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None], name='y-input')
        pred = self.inference(m, X, X_dim)
        global_steps = tf.Variable(0, trainable=False)

        variable_averages = tf.train.ExponentialMovingAverage(self.moving_average_decay, global_steps)
        variable_averages_op = variable_averages.apply(tf.trainable_variables())

        cross_entropy_mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y_))
        loss = tf.add_n([cross_entropy_mean])
        learning_rate = tf.train.exponential_decay(self.learning_rate_base,
                                                   global_steps,
                                                   self.data_length / self.batch_size,
                                                   self.learning_rate_decay)
        # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_steps)
        train_step = tf.train.FtrlOptimizer(learning_rate).minimize(loss, global_step=global_steps)
        with tf.control_dependencies([train_step, variable_averages_op]):
            train_op = tf.no_op(name='train')

        time_s = time.time()
        result = []
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in tqdm(range(n_epoch), unit='epoch', disable=False):
                # f_dict = {X: test_x, y_: test_y}
                f_dict = {X: train_x, y_: train_y}
                # print(sess.run(learning_rate))
                _, cost_, predict_ = sess.run([train_op, loss, pred], feed_dict=f_dict)
                auc = roc_auc_score(train_y, predict_)
                # print('train auc: ', auc)
                time_t = time.time()
                if epoch % 100 == 0:
                    f_dict = {X: test_x, y_: test_y}
                    predict_test = sess.run(pred, feed_dict=f_dict)
                #     # _, cost_, predict_test = sess.run([train_op, loss, pred], feed_dict=f_dict)
                #     # print("predict_test: ", predict_test)
                    test_auc = roc_auc_score(test_y, predict_test)
                    print("%d %ld cost:%f,train_auc:%f,test_auc:%f" % (epoch, (time_t - time_s), cost_, auc, test_auc))
                    result.append([epoch,(time_t - time_s),auc,test_auc])
                pd.DataFrame(result, columns=['epoch','time','train_auc','test_auc']).to_csv("data/mlr_"+str(m)+'.csv')
            saver.save(sess, os.path.join(self.model_save_path, self.model_save_name))

def main(argv=None):
    MLRClass().train()

if __name__ == "__main__":
    tf.app.run()