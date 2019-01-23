# encoding=utf-8

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from MLR import get_data, MLRClass
import os

class Eval_MLR_Model(object):
    def __init__(self):
        self.mlr = MLRClass()

    def evaluate(self, m=2):
        _, _, X_test, y_test = get_data()
        X_dim = X_test.shape[1]
        X = tf.placeholder(tf.float32, [None, X_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None], name='y-input')
        f_dict = {X: X_test, y_: y_test}
        pred = self.mlr.inference(m, X, X_dim)   # X 输入一个Tensor即可

        variable_averages = tf.train.ExponentialMovingAverage(self.mlr.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.mlr.model_save_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                predict_ = sess.run(pred, feed_dict=f_dict)
                test_auc = roc_auc_score(y_test, predict_)
                print('test auc: ', test_auc)
            else:
                print("no checkpoint file found.")
                return

def main(argv=None):
    Eval_MLR_Model().evaluate()

if __name__ == "__main__":
    tf.app.run()
