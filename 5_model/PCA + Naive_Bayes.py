import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
dataset = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8',delimiter=",")
# dataset = pd.read_csv('smote.csv', encoding='utf-8',delimiter=",")

dataset = dataset.values
X = dataset[:,:-7]
y1 = dataset[:,-7]
pca_model = PCA(n_components=88)
"""左奇异"""
X_new = pca_model.fit_transform(X)#这个是U×S×V中的U
""""右奇异"""
components = pca_model.components_#这个是U×S×V中的V
components = X.dot(components.T)
one = (y1!=0).sum()
zero = (y1==0).sum()
print(one)
print(zero)
iteration = [10,20,30,40,50,60,70,80,90,100]
s = 0
s_y = 0
for k in iteration:
    for i in range(k):
        X_train,X_test, y_train, y_test  = train_test_split(X_new,y1,test_size=0.2,)
        #X_train,X_test, y_train, y_test  = train_test_split(components,y1,test_size=0.2,)
        clf = GaussianNB()
        #clf = MultinomialNB()
        #clf = BernoulliNB()
        clf = clf.fit(X_train, y_train)
        y_pred=clf.predict(X_test)
        # print(y_test)
        # print(y_pred)
        # print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
        s += X_test.shape[0]
        s_y += (y_test != y_pred).sum()
    print(k,"——","错误率：",s_y/s)