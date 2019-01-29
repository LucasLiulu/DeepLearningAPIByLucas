# coding=utf-8

from MLR import MLRClass, get_data
from MLR_models import MLRClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# for m in tqdm([5, 10, 15, 20, 1], unit='m', disable=False):
#     MLRClass().train(m=m, n_epoch=200)

train_x, train_y, test_x, test_y = get_data()
data_length, n_features = train_x.shape
mlr = MLRClassifier(m=5, seed=12, n_features=n_features, data_length=data_length, batch_size=100, l2=0.0001)
print("mlr. l2_regularization_rate: ", mlr.core.l2_regularization_rate)
mlr.fit(train_x, train_y, show_progress=False, early_stopping_rounds=10, n_epochs=10000, verbose_eval=1, metric='auc', maximize=True)
print('mlr.best_score: ', mlr.best_score)
print("mlr.best_iteration: ", mlr.best_iteration)
print("mlr.best_score: ", mlr.best_score)
print("mlr. l2_regularization_rate: ", mlr.core.l2_regularization_rate)

pred = mlr.predict(test_x)
prob = mlr.predict_proba(test_x)
# print(pred)
# print(prob)
print('accuracy: ', accuracy_score(test_y, pred))
print('log_loss: ', log_loss(test_y, prob[:, 1]))
print('auc: ', roc_auc_score(test_y, prob[:, 1]))
mlr.save_state('model', 'model.ckpt')