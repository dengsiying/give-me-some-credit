# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:49:36 2018

@author: daisydeng
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",101)
pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法
pd.options.display.max_rows = 15 #最多显示15行
import warnings
warnings.filterwarnings('ignore') #为了整洁，去除弹出的warnings

df = pd.read_csv('data/cs-training.csv')
#print(df.describe())
#print(df.isnull().sum()) # 计算每个列的空值数目，MonthlyIncome和NumberOfDependents的缺失值分别为29731和3924。

######数据清洗
#删除第一列Unnamed
df = df.drop(df.columns[0],axis=1)
#查看违约率在每个自变量上的分布，生成频率分布表
#从RevolvingUtilizationOfUnsecuredLines开始
df_tmp = df[['SeriousDlqin2yrs','RevolvingUtilizationOfUnsecuredLines']]
#使用cut函数，将连续变量转换成分类变量
def binning(col, cut_points, labels=None,isright=True):
    minval = col.min()
    maxval = col.max()
    break_points = [minval] + cut_points + [maxval]
     
    if not labels:
        labels = range(len(cut_points)+1)
    else:
        labels=[str(i+1)+":"+labels[i] for i in range(len(cut_points)+1)]  
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True,right=isright)
    return colBin

cut_points = [0.25,0.5,0.75,1,2]
labels = ["below0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-2.0","above2"]
df_tmp['Utilization_Bin'] = binning(df_tmp["RevolvingUtilizationOfUnsecuredLines"], cut_points, labels)
#print(df_tmp['Utilization_Bin'])
#总人数
#total_size = df_tmp.shape[0]
#per_table=pd.pivot_table(df_tmp,
#                         index=['Utilization_Bin'], 
#                         aggfunc={'RevolvingUtilizationOfUnsecuredLines':[len, lambda x:len(x)/total_size*100],'SeriousDlqin2yrs':[np.sum]},
#                         values=['RevolvingUtilizationOfUnsecuredLines','SeriousDlqin2yrs']
#                         )
##print(per_table)
##print(per_table['RevolvingUtilizationOfUnsecuredLines','<lambda>'])
#per_table['SeriousDlqin2yrs','percent']=per_table['SeriousDlqin2yrs','sum']/per_table['RevolvingUtilizationOfUnsecuredLines','len']*100
#per_table=per_table.rename(columns={'<lambda>':'percent','len': 'number','sum':'number'})
#per_table=per_table.reindex_axis((per_table.columns[1],per_table.columns[0],per_table.columns[2],per_table.columns[3]),axis=1)
#print(per_table)
# 把上述生成频率表的过程写成函数，用于对每个自变量进行类似处理
def get_frequency(df,col_x,col_y, cut_points, labels,isright=True):
    df_tmp=df[[col_x,col_y]]
    df_tmp['columns_Bin']=binning(df_tmp[col_x], cut_points, labels,isright=isright)
    total_size=df_tmp.shape[0] 
    per_table=pd.pivot_table(df_tmp,index=['columns_Bin'], aggfunc={col_x:[len, lambda x:len(x)/total_size*100],col_y:[np.sum] },values=[col_x,col_y])
    if(per_table.columns[0][0]!=col_x): #假如col_x不在第一列，说明是在第2、3列，就把它们往前挪
        per_table=per_table.reindex_axis((per_table.columns[1],per_table.columns[2],per_table.columns[0]),axis=1)
    per_table[col_y,'percent']=per_table[col_y,'sum']/per_table[col_x,'len']*100
    per_table=per_table.rename(columns={'<lambda>':'percent','len': 'number','sum':'number'})
    per_table=per_table.reindex_axis((per_table.columns[1],per_table.columns[0],per_table.columns[2],per_table.columns[3]),axis=1)
    return per_table
#Age
cut_points=[25,35,45,55,65]
labels=['below25', '26-35', '36-45','46-55','56-65','above65']
feq_age=get_frequency(df,'age','SeriousDlqin2yrs', cut_points, labels)
#print(feq_age)
#DeptRatio
cut_points = [0.25,0.5,0.75,1,2]
labels = ["below0.25","0.25-0.5","0.5-0.75","0.75-1.0","1.0-2.0","above2"]
feq_ratio=get_frequency(df,'DebtRatio','SeriousDlqin2yrs', cut_points, labels)
#print(feq_ratio)
#NumberOfOpenCreditLinesAndLoans
cut_points=[5,10,15,20,25,30]
labels=['below 5', '6-10', '11-15','16-20','21-25','26-30','above 30']
feq_OpenCredit=get_frequency(df,'NumberOfOpenCreditLinesAndLoans','SeriousDlqin2yrs', cut_points, labels)
#print(feq_OpenCredit)
#NumberRealEstateLoansOrLines
cut_points=[5,10,15,20]
labels=['below 5', '6-10', '11-15','16-20','above 20']
feq_RealEstate=get_frequency(df,'NumberRealEstateLoansOrLines','SeriousDlqin2yrs', cut_points, labels)
#print(feq_RealEstate)
#NumberOfTime30-59DaysPastDueNotWorse
cut_points=[1,2,3,4,5,6,7]
labels=['0','1','2','3','4','5','6','7 and above',]
feq_30days=get_frequency(df,'NumberOfTime30-59DaysPastDueNotWorse','SeriousDlqin2yrs', cut_points, labels,isright=False)
#print(feq_30days)
#MonthlyIncome
cut_points=[5000,10000,15000]
labels=['below 5000', '5000-10000','1000-15000','above 15000']
feq_Income=get_frequency(df,'MonthlyIncome','SeriousDlqin2yrs', cut_points, labels)
#print(feq_Income)
#NumberOfDependents
cut_points = [1,2,3,4,5]
labels = ["0","1","2","3","4","5 and more"]
feq_dependent=get_frequency(df,'NumberOfDependents','SeriousDlqin2yrs', cut_points, labels,isright=False)
print(feq_dependent)




