# coding=utf-8
from MLR_core import MLRCore
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, log_loss
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

def batch_feeddict(X, y=None, w=None, core=None):
    if core is None:
        raise ValueError('must give a core!')
    fd = {}
    fd[core.X_train] = X.astype(np.float32)
    if y is not None:
        fd[core.y_train] = y.astype(np.float32)
    if w is not None:
        fd[core.w_train] = w.astype(np.float32)
    return fd

class MLRClassifier(MLRCore):
    def __init__(self, seed=None, session_config=None, n_epoches=100,
                 batch_size=-1, verbose_eval=0,
                 **core_arguments):
        core_arguments['seed'] = seed
        self.seed = seed
        self.core = MLRCore(**core_arguments)
        self.session_config = session_config
        self.n_epoches = n_epoches
        self.batch_size = batch_size
        self.steps = 0
        self.verbose_eval = verbose_eval
        self.early_stopping_rounds = 0    # 记录是否达到早停次数

    def fit(self, X, y, sample_weight=None, pos_class_weight=None, n_epochs=None, metric=None,
            show_progress=False, maximize=None, verbose_eval=None, early_stopping_rounds=None, l2=None):
        '''

        :param X:
        :param y:
        :param sample_weight: balanced or None
        :param pos_class_weight: array like or None
        :param n_epochs:
        :param show_progress:
        :param early_stopping_rounds:
        :return:
        '''
        if not (set(y) == set([0, 1])):
            raise ValueError('Input labels must be in set {0, 1}.')

        assert sample_weight is None or pos_class_weight is None, 'sample_weight and pos_class_weight are mutually exclusive parameters'
        if sample_weight:
            self.sample_weight = sample_weight
        if pos_class_weight:
            self.pos_class_weight = pos_class_weight
        used_w = self._preprocess_sample_weights(sample_weight, pos_class_weight, y)
        self._fit(X, y, used_w, n_epochs, show_progress, maximize=maximize, metric=metric,
                  verbose_eval=verbose_eval, early_stopping_rounds=early_stopping_rounds, l2=l2)

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

    def _fit(self, X_, y_, w_, n_epochs=None, show_progress=False, maximize=False,
             verbose_eval=None, early_stopping_rounds=None, metric='logloss', l2=None):
        '''
        训练模型的核心类。
        其中包括模型的训练流程和早停。
        :param X_: 输入训练数据，需要是pandas DataFrame格式
        :param y_: 训练数据对应的标签，同样需要是pandas Dataframe格式
        :param w_: 不平衡样本权重，是numpy 数组，数组元素是float类型
        :param n_epochs:训练轮数，int类型
        :param show_progress:是否显示进度条，bool类型
        :param maximize:对于默写评价指标，是变小好还是变大好，bool类型。如果是变大好，则为TRUE；否则为FALSE
        :param verbose_eval:每训练多少轮打印一次评价指标的值。int类型
        :param early_stopping_rounds:这么多轮训练模型的指标都没有变好，则停止训练
        :param metric:训练过程中使用的评价指标，可用于早停
        :return:
        '''
        if self.core.n_features is None:
            self.core.set_num_features(X_.shape[1])
        assert self.core.n_features == X_.shape[1], 'Different num of features in initialized graph and input'

        if self.core.graph is None:
            self.core.build_graph()
            self.init_session()

        if n_epochs is None:
            n_epochs = self.n_epoches

        if self.seed:
            np.random.seed(self.seed)

        if verbose_eval:
            self.verbose_eval = verbose_eval

        # 添加早停
        if early_stopping_rounds:
            earlyStopClass = EarlyStopClass(maximize_score=maximize,
                                            stopping_rounds=early_stopping_rounds,
                                            verbose=bool(verbose_eval),
                                            metric=metric)
        break_epoch = False    # 用于早停
        metric_dict = {'auc': roc_auc_score, 'logloss': log_loss}    # 评价函数
        for epoch in tqdm(range(n_epochs), unit='epoch', disable=(not show_progress)):
            if break_epoch:
                break
            # 返回一个洗牌后的矩阵副本
            perm = np.random.permutation(X_.shape[0])
            epoch_loss = []
            for bX, by, bw in batcher(X_.iloc[perm], y_.iloc[perm], w_[perm], batch_size=self.batch_size):
                fd = batch_feeddict(bX, y=by, w=bw, core=self.core)
                ops_to_run = [self.core.trainer_op, self.core.loss, self.core.outputs]
                _, batch_loss, predict = self.session.run(ops_to_run, feed_dict=fd)
                epoch_loss.append(batch_loss)
                self.steps += 1

            # 训练时的模型评价指标
            valided_fd = batch_feeddict(X_, y=y_, w=w_, core=self.core)
            batch_loss_, predict_ = self.session.run([self.core.loss, self.core.outputs], feed_dict=valided_fd)
            if metric == 'logloss':
                metric_value = np.mean(batch_loss)
            else:
                metric_value = metric_dict[metric](y_, predict_)
            if self.verbose_eval:
                if epoch % self.verbose_eval == 0:
                    print('[epoch {}]: metric {} value: {}'.format(epoch, metric, metric_value))

            if early_stopping_rounds:
                evaluation_result = metric_value
                if earlyStopClass.callback(epoch, evaluation_result):
                    print('\nearly_stopping_rounds: ', early_stopping_rounds)
                    self.best_score = earlyStopClass.best_score
                    self.best_iteration = earlyStopClass.best_iteration
                    self.best_msg = earlyStopClass.best_msg
                    break_epoch = True

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


class EarlyStopClass(object):
    def __init__(self, stopping_rounds=None, verbose=None, metric=None, maximize_score=None):
        self.best_msg = ''
        self.best_score = float(np.inf)
        self.best_iteration = float(np.inf)
        self.stopping_rounds = stopping_rounds
        self.verbose = verbose    # True or False
        self.maximize_score = maximize_score     # 用于指明指标优化方向，是往小了优化还是往大了优化，往大优化为TRUE

        msg = ("Multiple eval metrics have been passed: "
               "'{0}' will be used for early stopping.\n\n")
        print(msg.format(metric))
        msg = "Will train until {} hasn't improved in {} rounds.\n"
        print(msg.format(metric, stopping_rounds))


    def callback(self, iteration=0, evaluation_result=0):
        best_score = self.best_score * (-1) if (self.best_score is np.inf and self.maximize_score) else self.best_score
        best_iteration = self.best_iteration
        if (self.maximize_score and evaluation_result > best_score) or (not self.maximize_score and evaluation_result < best_score):
            msg = '[%d]\t%s' % (iteration, '\t'.join(evaluation_result)) if isinstance(evaluation_result, list) \
                else '[%d]\t%s' % (iteration, '\t' + str(evaluation_result))
            self.best_msg = msg
            self.best_score = evaluation_result
            self.best_iteration = iteration
            return False
        elif iteration - best_iteration >= self.stopping_rounds:
            best_msg = self.best_msg
            if self.verbose:
                msg = "Stopping. Best iteration: \n{}\n\n"
                print(msg.format(best_msg))
            return True

