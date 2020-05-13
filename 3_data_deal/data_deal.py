import pandas as pd
import numpy as np


"""数据读取"""
all_data = pd.read_csv('../2_data_merge/all_data.csv',encoding = "utf-8")

"""
直接删除无关特征
有的是日期格式的，有的大量缺失，有的是条码，
"""
all_data_new = all_data.drop(labels = ['序列号','姓名','病案号','ID号','联系电话','血型','诊断','并发症','术前条码_x',
    '术后条码_x','POD1条码_x','POD2条码_x','POD3条码_x','POD4条码_x','POD5条码_x','POD6条码_x',
    'POD7条码_x','POD14条码_x','术前条码_y','术后条码_y','POD1条码_y','POD2条码_y','POD3条码_y',
    'POD4条码_y','POD5条码_y','POD6条码_y','POD7条码_y','POD14条码_y','门脉开放时间_x','术前时间点',
    '开放前时间点','距开放时间-0(min)','开放10min内','距开放时间-10(min)','开放30min内',
    '距开放时间-30(min)','开放30-60min','距开放时间-60(min)','开放60-90min内','距开放时间-90(min)',
    '开放90-120min内','距开放时间-120(min)','开放120-150min内','距开放时间-150(min)','开放150-180min内',
    '距开放时间-180(min)','开放180-210min内','距开放时间-210(min)','开放210-240min内','距开放时间-240(min)',
    '术毕','距开放时间-end(min)','入ICU','距开放时间-icu(min)','术后1天','距开放时间-1d(min)',
    '距开放时间-1d(h)','术后2天','距开放时间-2d(min)','距开放时间-2d(h)','术前条码','术后条码','POD1条码',
    'POD2条码','POD3条码','POD4条码','POD5条码','POD6条码','POD7条码','POD14条码','手术开始日期_x',
    '手术结束日期','门脉阻断时间','门脉开放时间_y','手术开始日期_y','POD0日期','最后记录日期','最后记录天数d',
    '手术结束时刻','拔管时刻','术后带管时间h','ICU驻留时间d','术后住院时间d','术后住院转归(1康复出院2死亡3自动离院)',
    '术后死亡时间d','术后自动离院时间d','死亡或放弃治疗原因','最后记录状态','自动离院原因','术后30天存活1死亡2',
    '术后90天存活1死亡2','术后180天存活1死亡2','术后1年存活1死亡2','出院后情况','具体情况.3','具体情况.9'
                                       ], axis=1,inplace=True)

column_headers = list(all_data.columns.values)
print("数据列名称：",column_headers)
for column in column_headers:
    print(all_data[column].isnull().any(axis=0))

b_feature = ['具体情况_x','间隔时间','18其它','具体情况_y','具体情况.1','具体情况.2',
                '具体情况.4','具体情况.5','具体情况.6','具体情况.7','具体情况.8','其它并发症']

for column in column_headers:
    if column in b_feature:
        continue
    if all_data[column].isnull().sum(axis = 0)/all_data.shape[0] > 0.8:
        all_data.drop(labels=column, axis = 1, inplace=True)
    else:
        if all_data[column].mean() >0.5 and all_data[column].mean() < 1:
            mean = 1
            all_data[column].fillna(mean,inplace=True)
        elif all_data[column].mean() > 0 and all_data[column].mean() <= 0.5:
            mean = 0
            all_data[column].fillna(mean,inplace=True)
        else:
            all_data[column].fillna(all_data[column].mean(), inplace=True)

# 替换法处理缺失值
all_data.fillna(value = {'热缺血时间min': all_data['热缺血时间min'].mode()[0],
                         '术后并发症IV': all_data['术后并发症IV'].mode()[0], # 使用变量的众数替换缺失
                 '冷缺血时间h':all_data['冷缺血时间h'].mean(), # 使用变量的平均值替换缺失
              '4%白蛋白':all_data['4%白蛋白'].mean(),'出血量':all_data['出血量'].mean(),
              'I期尿量':all_data['I期尿量'].mean(),'II期尿量':all_data['II期尿量'].mean(),
             'III期尿量':all_data['III期尿量'].mean(),'特利加压素ml/h': all_data['特利加压素ml/h'].mode()[0],
         '氨甲环酸g/h':all_data['氨甲环酸g/h'].mean(),'肝肾联合移植':all_data['肝肾联合移植'].mean()
                         },
          inplace = True # 原地修改数据
          )

for column in b_feature:
    if all_data[column].isnull().any(axis=0):
        # print('=============================================')
        # print('当前特征：', column)
        # print(all_data[column].value_counts())
        all_data[column].fillna(0,inplace=True)

all_data.to_csv('./all_data_new.csv'.format(), sep=',', index=False)

dataset = pd.read_csv('all_data_new.csv', encoding='utf-8',delimiter=",")
drop_feature = ['具体情况_x','间隔时间','18其它','具体情况_y','具体情况.1','具体情况.2',
                '具体情况.4','具体情况.5','具体情况.6','具体情况.7','具体情况.8','其它并发症']
for feature in drop_feature:
    print('---------------------')
    print('查看当前特征列：',feature)
    size = len(dataset[feature].unique())
    print('长度：',size)
    cur = []
    for i in range(size):
        cur.append(i)
    if feature == '其它并发症':
        cur[0],cur[1] = cur[1],cur[0]
    # print('cur',cur)
    cur_data = pd.Categorical(dataset[feature],dataset[feature].unique())
    dataset[feature] = cur_data.rename_categories(cur)
print(dataset)
dataset.to_csv('./all_data_last.csv'.format(), sep=',', index=False)
