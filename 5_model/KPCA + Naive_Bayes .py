
import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
np.seterr(divide='ignore', invalid='ignore')
#df = pd.read_csv('../5_model/smote.csv', encoding='utf-8',delimiter=",")
df = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8', delimiter=",")
df = df.values
"""数据"""
X = df[:, :-7]
print("原始数据：",X.shape)

"""标签"""
Y = df[:, -7]
kpca = KernelPCA(kernel='rbf',gamma=10,n_components=100)
X_new = kpca.fit_transform(X)
k = 100
num_sum = 0
acc = 0
for i in range(k):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(X_new, Y, test_size=0.2, )
    clf = GaussianNB()
    # clf = MultinomialNB()
    # clf = BernoulliNB()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
    acc += (y_test != y_pred).sum()
    num_sum += X_test.shape[0]
print('错误率为：',acc/num_sum)