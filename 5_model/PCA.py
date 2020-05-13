import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
"""数据导入"""
df = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8', delimiter=",")
df = df.values

"""数据"""
X = df[:, :-7]
print("原始数据：",X.shape)

"""标签"""
Y = df[:, -7]
# 使用sklearn的PCA进行维度转换
# 建立PCA模型对象 n_components控制输出特征个数
pca_model = PCA(n_components=42)
# 将数据集输入模型
#pca_model.fit(X)
#X_pca = pca_model.fit_transform(X)
# 对数据集进行转换映射
#components = pca_model.transform(X)
X_new = pca_model.fit_transform(X)#这个是U×S×V中的U
#X_new = X.dot(X_new.T)
# 获得转换后的所有主成分
components = pca_model.components_#这个是U×S×V中的V
# components_old = pca_model.components_#这个是U×S×V中的V
# print(X.shape)
# print(components_old.shape)
# components = X.dot(components_old.T)
print("PCA降维后的数据：",X_new.shape)
print("标签：",Y.shape)
print("最大方差的维度：",components.shape)
#print(components)
# 获得各主成分的方差
components_var = pca_model.explained_variance_
# 获取主成分的方差占比
components_var_ratio = pca_model.explained_variance_ratio_
# 打印方差
print('方差：',np.round(components_var,3))
# [3.002 1.659 0.68 ]
# 打印方差占比
print(np.round(components_var_ratio,3))

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
# dataset = pd.DataFrame(X_new)
# data = dataset.corr()  #test_feature => pandas.DataFrame#
# sns.heatmap(data)
# plt.show()
