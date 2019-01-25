# coding=utf-8
from MLR_core import MLRCore
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
import tensorflow as tf
import numbers, sklearn, os


def batcher(X_, y_=None, w_=None, batch_size=-1):
    '''
    将输入数据分成 batch 大小的小数据集
    :param y_:
    :param w_:
    :param batch_size:
    :return:
    '''
    n_samples = X_.shape[0]
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i: upper_bound]
        ret_y = None
        ret_w = None
        if y_ is not None:
            ret_y = y_[i: upper_bound]
        if w_ is not None:
            ret_w = w_[i: upper_bound]
        yield (ret_x, ret_y, ret_w)

def batch_feeddict(X, y, w, core):
    fd = {}
    fd[core.X_train] = X.astype(np.float32)
    if y is not None:
        fd[core.y_train] = y.astype(np.float32)
    if w is not None:
        fd[core.w_train] = w.astype(np.float32)
    return fd

class MLRClassifier(MLRCore):
    def __init__(self, seed=None, session_config=None, n_epoches=100,
                 batch_size=-1, verbose=0,
                 **core_arguments):
        core_arguments['seed'] = seed
        self.seed = seed
        self.core = MLRCore(**core_arguments)
        self.session_config = session_config
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.steps = 0
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, pos_class_weight=None, n_epochs=None, show_progress=False):
        if not (set(y) == set([0, 1])):
            raise ValueError('Input labels must be in set {0, 1}.')

        assert sample_weight is None or pos_class_weight is None, 'sample_weight and pos_class_weight are mutually exclusive parameters'
        if sample_weight:
            self.sample_weight = sample_weight
        if pos_class_weight:
            self.pos_class_weight = pos_class_weight
        used_w = self._preprocess_sample_weights(sample_weight, pos_class_weight, y)
        self._fit(X, y, used_w, n_epochs, show_progress)

    '''
    根据正负样本比例调整样本权重
    '''
    def _preprocess_sample_weights(self, sample_weight, pos_class_weight, used_y):
        used_w = np.ones_like(used_y)
        if sample_weight is None and pos_class_weight is None:
            return used_w
        if isinstance(pos_class_weight, numbers.Number):
            used_w[used_y > 0] = pos_class_weight
        elif sample_weight == 'balanced':
            pos_rate = np.mean(used_y > 0)
            neg_rate = 1 - pos_rate
            used_w[used_y > 0] = neg_rate / pos_rate
            used_w[used_y < 0] = 1.0
        elif isinstance(sample_weight, np.ndarray) and len(sample_weight.shape) == 1:
            used_w = sample_weight
        else:
            raise ValueError("Unexpected type for sample_weight or pos_class_weight parameters.")
        return used_w

    def init_session(self):
        '''
        Start computational session on builded graph
        :return:
        '''
        if self.core.graph is None:
            raise ('Graph not found. Try call .core.build_graph() before .initialize_session()')
        self.session = tf.Session(config=self.session_config, graph=self.core.graph)
        self.session.run(self.core.init_all_vars)

    def _fit(self, X_, y_, w_, n_epochs=None, show_progress=False):
        if self.core.n_features is None:
            self.core.set_num_features(X_.shape[1])
        assert  self.core.n_features == X_.shape[1], 'Different num of features in initialized graph and input'

        if self.core.graph is None:
            self.core.build_graph()
            self.init_session()

        if n_epochs is None:
            n_epochs = self.n_epoches

        if self.seed:
            np.random.seed(self.seed)

        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            # 返回一个洗牌后的矩阵副本
            # print('X_ shape: ', X_.shape)
            perm = np.random.permutation(X_.shape[0])
            epoch_loss = []
            for bX, by, bw in batcher(X_.iloc[perm], y_.iloc[perm], w_[perm], batch_size=self.batch_size):
                fd = batch_feeddict(bX, by, bw, core=self.core)
                ops_to_run = [self.core.trainer_op, self.core.loss, self.core.outputs]
                _, batch_loss, predict = self.session.run(ops_to_run, feed_dict=fd)
                epoch_loss.append(batch_loss)
                self.steps += 1

            if self.verbose:
                print('[epoch {}]: mean loss value: {}'.format(epoch, np.mean(epoch_loss)))

            # print('predict shape: ', predict.shape, 'predict: ', predict)
            # print("y_ iloc perm shape: ", y_.iloc[perm])
            # print('auc: ', roc_auc_score(y_.iloc[perm], predict))

    def decision_function(self, X, pred_batch_size=None):
        if self.core.graph is None:
            raise sklearn.exceptions.NotFittedError("Call fit before prediction")
        output = []
        if pred_batch_size is None:
            pred_batch_size = self.batch_size

        for bX, by, bw in batcher(X, y_=None, batch_size=pred_batch_size):
            fd = batch_feeddict(bX, by, bw, core=self.core)
            output.append(self.session.run(self.core.outputs, feed_dict=fd))
        distances = np.concatenate(output).reshape(-1)
        return distances

    def predict(self, X, pred_batch_size=None):
        raw_output = self.decision_function(X, pred_batch_size)
        predictions = (raw_output > 0.5).astype(int)
        return predictions

    def predict_proba(self, X, pred_batch_size=None):
        outputs = self.decision_function(X, pred_batch_size)
        probs_positive = outputs
        probs_negative = 1 - probs_positive
        probs = np.vstack((probs_negative.T, probs_positive))
        return probs.T

    def save_state(self, path, name):
        self.core.saver.save(self.session, os.path.join(path, name))

    def load_state(self, path):
        if self.core.graph is None:
            self.core.build_graph(restore=True)
            self.init_session()
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.core.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            raise ('no checkpoint file found.')

    def destory(self):
        self.session.close()
        self.core.graph = None

