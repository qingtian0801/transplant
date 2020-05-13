from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB


dataset = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8',delimiter=",")
dataset = dataset.values
X = dataset[:,0:-7]
y1 = dataset[:,-7]
one = (y1!=0).sum()
zero = (y1==0).sum()
print(one)
print(zero)
k = 100
num_sum = 0
acc = 0
for i in range(k):
    X_train,X_test, y_train, y_test  = train_test_split(X,y1,test_size=0.2,)
    clf = GaussianNB()
    #clf = MultinomialNB()
    #clf = BernoulliNB()
    clf = clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_test)
    print(y_pred)
    print("高斯朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (X_test.shape[0],(y_test != y_pred).sum()))
    acc += (y_test != y_pred).sum()
    num_sum += X_test.shape[0]
print('错误率为：',acc/num_sum)