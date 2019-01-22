# encoding=utf-8

import tensorflow as tf
import pandas as pd
import numpy as np
import time, os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

def get_data():
    train_data = pd.read_csv("data\\adult.data.txt",header=None,delimiter=',', encoding='GB2312')
    test_data = pd.read_csv("data\\adult.test.txt",header=None,delimiter=',', encoding='GB2312')

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
        self.feature_length = 0
        self.data_length = 0
        self.model_save_path = 'model'
        self.model_save_name = 'model.ckpt'
        self.graph = None

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

    def train(self, m=2, learning_rate=0.1, n_epoch=1000):
        train_x, train_y, test_x, test_y = get_data()
        X_dim = train_x.shape[1]
        X = tf.placeholder(tf.float32, [None, X_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None], name='y-input')

        pred = self.inference(m, X, X_dim)
        cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y_))
        cost = tf.add_n([cost1])
        train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)

        time_s = time.time()
        result = []
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_epoch):
                f_dict = {X: train_x, y_: train_y}
                _, cost_, predict_ = sess.run([train_op, cost, pred], feed_dict=f_dict)
                auc = roc_auc_score(train_y, predict_)
                time_t = time.time()
                if epoch % 100 == 0:
                    f_dict = {X: test_x, y_: test_y}
                    _, cost_, predict_test = sess.run([train_op, cost, pred], feed_dict=f_dict)
                    test_auc = roc_auc_score(test_y, predict_test)
                    print("%d %ld cost:%f,train_auc:%f,test_auc:%f" % (epoch, (time_t - time_s), cost_, auc, test_auc))
                    result.append([epoch,(time_t - time_s),auc,test_auc])
                pd.DataFrame(result,columns=['epoch','time','train_auc','test_auc']).to_csv("data/mlr_"+str(m)+'.csv')
            saver.save(sess, os.path.join(self.model_save_path, self.model_save_name))

def main(argv=None):
    MLRClass().train()

if __name__ == "__main__":
    tf.app.run()