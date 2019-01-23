# coding=utf-8

from MLR import MLRClass, get_data
from MLR_models import MLRClassifier
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

# for m in tqdm([5, 10, 15, 20, 1], unit='m', disable=False):
#     MLRClass().train(m=m, n_epoch=200)

train_x, train_y, test_x, test_y = get_data()
data_length, n_features = train_x.shape
mlr = MLRClassifier(m=5, seed=12, n_features=n_features, data_length=data_length, batch_size=100)
mlr.fit(train_x, train_y, show_progress=True)
pred = mlr.predict(test_x)
prob = mlr.predict_proba(test_x)
print('accuracy: ', accuracy_score(test_y, pred))
print('auc: ', roc_auc_score(test_y, prob[:, 1]))