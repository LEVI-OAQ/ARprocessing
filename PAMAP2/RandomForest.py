"""
    使用随机森林进行活动识别

    读入,整合数据（经预处理、信号处理和特征提取） from D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe/subject1xx.dat
    读入特征名 from D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe/feature_names.txt
    训练集，测试集划分
    定义模型
    训练
    测试
    输出结果
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier as rfc


# fe_data_dir = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe_AccTransfer/'
fe_data_dir = 'D:/pyProjects/ARprocessing/PAMAP2/Protocol/fe/'
filenames = os.listdir(fe_data_dir)
with open(os.path.join(fe_data_dir, 'feature_names.txt'), 'r') as f:
    feature_names = [name[0:-1] for name in f.readlines()]
print(f'total features: {len(feature_names)}')
data = np.empty((0, len(feature_names) + 1))
for file in filenames:
    if file.endswith('.dat'):
        print(f'{file} is loading')
        tp_data = np.loadtxt(os.path.join(fe_data_dir, file))
        data = np.vstack((data, tp_data))

data = pd.DataFrame(data)
print('data shape:', data.shape)
train = data.sample(frac=0.8, random_state=0)
test = data[~data.index.isin(train.index)]

x_train = train[train.columns[:-1]]
y_train = train[train.columns[-1]]

x_test = test[test.columns[:-1]]
y_test = test[test.columns[-1]]

clf = rfc(n_estimators=50,
              random_state=0,
              oob_score=True)

t0 = time()
model = clf.fit(x_train, y_train)
train_time = time() - t0
print('train time %0.3f' % train_time)
print('oob_score: ', model.oob_score_)

t0 = time()
pred = clf.predict(x_test)
test_time = time() - t0
print('test time %0.3fs' % test_time)

score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3fs" % score)


