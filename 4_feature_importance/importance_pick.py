import pandas as pd
from xgboost import XGBClassifier


"""提取特征的index索引值"""
def index(impor_feature):
    num = []
    index = []
    k = 0
    for i in impor_feature:
        if i != 0:
            num.append(i)
            index.append(k)
        k += 1
    return index

"""合并所有的index索引，取其并集"""
def get_index(index1,index2):
    for i in index2:
        if i not in index1:
            index1.append(i)
    #return index1


dataset = pd.read_csv('../3_data_deal/all_data_last.csv', encoding='utf-8',delimiter=",")
dataset = dataset.values
"""y1为其它并发症"""
X = dataset[:,0:-30]
y1 = dataset[:,-1]
model = XGBClassifier()
model.fit(X, y1)
index1 = index(model.feature_importances_)
"""y2为V级(死亡)"""
X = dataset[:,0:-30]
y2 = dataset[:,-4]
model = XGBClassifier()
model.fit(X, y2)
index2 = index(model.feature_importances_)
"""y3为术后并发症IV"""
X = dataset[:,0:-30]
y3 = dataset[:,-8]
model = XGBClassifier()
model.fit(X, y3)
index3 = index(model.feature_importances_)
"""y4为术后并发症IIIb"""
X = dataset[:,0:-30]
y4 = dataset[:,-12]
model = XGBClassifier()
model.fit(X, y4)
index4 = index(model.feature_importances_)
"""y5为术后并发症IIIa"""
X = dataset[:,0:-30]
y5 = dataset[:,-17]
model = XGBClassifier()
model.fit(X, y5)
index5 = index(model.feature_importances_)
"""y6为术后并发症II"""
X = dataset[:,0:-30]
y6 = dataset[:,-25]
model = XGBClassifier()
model.fit(X, y6)
index6 = index(model.feature_importances_)
"""y7为术后并发症I"""
X = dataset[:,0:-30]
y7 = dataset[:,-30]
model = XGBClassifier()
model.fit(X, y7)
index7 = index(model.feature_importances_)

index_sum = [index1,index2,index3,index4,index5,index6,index7]

cur = []
for i in index_sum:
    get_index(cur,i)
cur.sort()
#print(cur)
data = pd.read_csv('../3_data_deal/all_data_last.csv', encoding='utf-8',delimiter=",")

column = data.columns.values.tolist()
print(len(column))
k = 0
feature_col = []
for num in column:
    if k in cur:
        feature_col.append(num)
    k += 1
#print(feature_col)
new = pd.DataFrame(data['其它并发症'])
for feature in feature_col:
    new[feature] = data[feature]
"""添加并发症标签"""
y_feature = [-30,-25,-17,-12,-8,-4,-1]
y_column = []
for num in y_feature:
    y_column.append(column[num])
for feature in y_column:
    new[feature] = data[feature]

drop = new['其它并发症']
del new['其它并发症']
new['其它并发症'] = drop
column1 = new.columns.values.tolist()
print(len(column1))
#new.drop(labels='Unnamed: 0')
new.to_csv('./final.csv'.format(), sep=',', index=False)