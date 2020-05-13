import pandas as pd


def xlsx_to_csv_pd(name):
    data_xls = pd.read_excel('liver transplantation.xlsx', index_col=0,sheet_name=name)
    #print(data_xls)
    data_xls.to_csv(name+'.csv', encoding='utf-8')


if __name__ == '__main__':
    name_list = ['一般情况',  '血常规', '生化', '血气', '凝血','术中情况',
                 '术中化验', '术后输血情况', '总体转归','出院前术后转归']
    for name in name_list:
        xlsx_to_csv_pd(name)