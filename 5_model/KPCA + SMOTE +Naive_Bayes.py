import pandas as pd
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
np.seterr(divide='ignore', invalid='ignore')
df = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8', delimiter=",")
df = df.values
"""数据"""
X = df[:, :-7]
print("原始数据：",X.shape)

"""标签"""
Y = df[:, -7]
"""KPCA的效果在26和22的时候最好"""
kpca = KernelPCA(kernel='rbf',gamma=10,n_components=100)
X_new = kpca.fit_transform(X)
dataset = pd.DataFrame(X_new)
print(dataset.shape)
#print(data)
dataset['label'] = Y
#print(data)

class Smote:
    #samples的最后一列是类标，都是1
    def __init__(self, samples, N=10,k=5):
        # 数据的行数和列数，即n_samples=样本数，n_attrs=样本的特征数
        self.n_samples, self.n_attrs=samples.shape
        # N=获取百分之多少的新增数据
        self.N=N
        # k=获取的最近邻数
        self.k=k
        # 原始的样本数据
        self.samples=samples

    def over_sampling(self):
        if self.N<100:
            # 原始数据的拷贝
            old_n_samples=self.n_samples
            print("old_n_samples", old_n_samples)
            # 获取多少个新增数据
            self.n_samples=int(float(self.N)/100*old_n_samples)
            print("n_samples", self.n_samples)
            # permutation()随机排列数组，多维数组按照第一个坐标轴的索引排列
            # 排列完之后取前n_samples个索引，n_sapmles已经是要新增的数据数了
            keep=np.random.permutation(old_n_samples)[:self.n_samples]
            print("keep", keep)
            # 获取原始数据的随机索引的数据
            new_samples=self.samples[keep]
            print("new_samples", new_samples)
            # 将数据赋值给samples
            self.samples=new_samples
            print("self.samples", self.samples)
            # N = 100，至此数据参数处理结束
            self.N=100
        # for i in range(len(self.samples)):
        #     print("=======================================")
        #     print(self.samples[i])
        N=int(self.N/100) #每个少数类样本应该合成的新样本个数
        # 生成一个n_samples行，n_attrs列的数组
        self.synthetic=np.zeros((self.n_samples*N, self.n_attrs))
        self.new_index=0
        # 获取数据的前k个近邻
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print("neighbors", neighbors)
        for i in range(len(self.samples)):
            # 获取第i个数据的k个邻居的下标
            # print("======================================")
            # print(self.samples[i])
            nnarray=neighbors.kneighbors([self.samples[i]],return_distance=False)[0]
            #存储k个近邻的下标
            # 此时N=1
            self.__populate(N, i, nnarray )
        return self.synthetic

    #从k个邻居中随机选取N次，生成N个合成的样本，此时N=1
    def __populate(self, N, i, nnarray):
        # 在这里N=1，即每次生成一个合成数据
        for j in range(N):
            # 生成0-（k-1）之间的随机数。
            nn = np.random.randint(0, self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]    #包含类标
            # 生成一行n_attrs列的数组，且服从某种分布的随机数
            gap=np.random.rand(1,self.n_attrs)
            # flatten()将数组降到一维
            self.synthetic[self.new_index]=self.samples[i]+gap.flatten()*dif
            self.new_index+=1

"""SMOTE进行数据扩充"""
# 设置数据扩充的次数
k = 10

for i in range(k):

    # X = data[:, :-1]
    # Y = data[:, -1]
    dataset = np.array(dataset)
    # print(dataset)
    smote = Smote(dataset, N=10, k=5)
    smote.over_sampling()
    #print('==========================================')
    #print(smote.synthetic)
    new = pd.DataFrame(smote.synthetic)
    new.to_csv('KPCA.csv'.format(), sep=',', mode='a', index=False,header=False)
data = pd.read_csv('../4_feature_importance/final.csv', encoding='utf-8', delimiter=",")

data_deal = pd.read_csv('KPCA.csv', encoding='utf-8',delimiter=",",error_bad_lines=False)
data_deal = data_deal.values
col = len(data_deal[0])
row = len(data_deal)
print("数据的行数：",row)
print("数据的列数：",col)
# data = dataset[88:,-7:]
# row,col = data.shape
index = [0,4,5,6,7,8,9,10,11]
for i in range(row):
    for j in index:
        for k in range(100):
            if data_deal[i][j] > k:
                continue
            else:
                if data_deal[i][j] >= (k - 0.5):
                    data_deal[i][j] = k
                    break
                else:
                    data_deal[i][j] = k - 1
                    break
    for j in range(col-7,col):
        if j == (col-1):
            for k in range(38):
                if data_deal[i][j] > k:
                    continue
                else:
                    if data_deal[i][j] >= (k-0.5):
                        data_deal[i][j] = k
                        break
                    else:
                        data_deal[i][j] = k-1
                        break
        else:
            if data_deal[i][j] >= 0.5:
                print("==========================")
                data_deal[i][j] = 1
            else:
                data_deal[i][j] = 0
#print(dataset[row-88][col-7] > 0)
data_deal = pd.DataFrame(data_deal)
#data_deal.columns = data.columns
frames = [data,data_deal]
result = pd.concat(frames)
result.to_csv('./KPCA_SMOTE.csv'.format(), sep=',', index=False)


data_smote = pd.read_csv('../5_model/KPCA_SMOTE.csv', encoding='utf-8',delimiter=",")
# data_smote.dropna(inplace=True)
print(data_smote.isnull().any())
data_smote = data_smote.values
X = data_smote[:,:-1]
y1 = data_smote[:,-1]
one = (y1!=0).sum()
zero = (y1==0).sum()
print(one)
print(zero)
iteration = [10,20,30,40,50,60,70,80,90,100]
s = 0
s_y = 0
# print(data_smote)
# print(np.isnan(data_smote).any())

for k in iteration:
    for i in range(k):
        X_train,X_test, y_train, y_test  = train_test_split(X,y1,test_size=0.2,)
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