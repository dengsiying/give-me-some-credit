# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:58:44 2018

@author: daisydeng
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer

from sklearn.linear_model.logistic import LogisticRegression
from sklearn import tree   

import warnings
warnings.filterwarnings('ignore') 
#RF模型：
#1.加载数据（训练和测试）和预处理数据
#2.将训练数据分解为training_new和test_new（用于验证模型）
#3.用Imputer处理数据：用Mean代替缺失值
#4.使用training_new数据建立RF模型：
#a 处理不平衡的数据分布
#b 使用带有CrossValidation的网格搜索执行参数调整
#c 输出最佳模型并对测试数据进行预测

#创建字典函数
#input: keys =[]and values=[]
#output: dict{}
def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    return lookup
#计算AUC函数
# input: y_true =[] and y_score=[]
# output: auc
def computeAUC(y_true,y_score):
    auc = roc_auc_score(y_true,y_score)
    print("auc=",auc)
    return auc

def main():
    #1，加载数据（训练和测试）和预处理数据
    #将NumberTime30-59，60-89，90中标记的96，98替换为NaN
    #将Age中的0替换为NaN
    colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59', 
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90', 
                'NORealEstate', 'NOTime60-89', 'NODependents']
    col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', [98, 96], 'NA', [98, 96], 'NA']
    col_na_values = creatDictKV(colnames, col_nas)
    dftrain = pd.read_csv("data\cs-training.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    #print(dftrain)
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv("data\cs-test.csv", names=colnames, na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()
    #2，使用StratifiedShuffleSplit将训练数据分解为training_new和test_new（用于验证模型）
    sss = StratifiedShuffleSplit(n_splits=1,test_size=0.33333,random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new
    #3，使用Imputer将NaN替换为平均值
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    x_test = imp.transform(x_test)
    #4，使用training_new数据建立RF模型：
    #a.设置rf的参数class_weight="balanced"为"balanced_subsample"
    #n_samples / (n_classes * np.bincount(y))
    rf = RandomForestClassifier(n_estimators=100,
                                oob_score= True,
                                min_samples_split=2,
                                min_samples_leaf=50,
                                n_jobs=-1,
                                class_weight='balanced_subsample',
                                bootstrap=True)
    #模型比较
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predicted_probs_train = lr.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    
    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    
    model = tree.DecisionTreeClassifier()    
    model.fit(x_train, y_train)
    predicted_probs_train = model.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    
    predicted_probs_test_new = lr.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    
    #输出特征重要性评估
    rf.fit(x_train, y_train)
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), dftrain.columns),reverse=True))
#    importances = rf.feature_importances_
#    indices = np.argsort(importances)[::-1]
#    feat_labels = dftrain.columns
#    for f in range(x_train.shape[1]):
#        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))
    #b.使用具有CrossValidation的网格搜索执行参数调整
    param_grid = {"max_features": [2, 3, 4], "min_samples_leaf":[50]}
    grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False)
    #c.输出最佳模型并对测试数据进行预测
    #使用最优参数和training_new数据构建模型
    grid_search.fit(x_train, y_train)
    print("the best parameter:", grid_search.best_params_)
    print("the best score:", grid_search.best_score_)

    #使用训练的模型来预测train_new数据
    predicted_probs_train = grid_search.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)
    #使用训练的模型来预测test_new数据（validataion data）
    predicted_probs_test_new = grid_search.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)
    #使用该模型预测test data
    predicted_probs_test = grid_search.predict_proba(x_test)
    predicted_probs_test = ["%.9f" % x[1] for x in predicted_probs_test]
    submission = pd.DataFrame({'Id':test_id, 'Probability':predicted_probs_test})
    submission.to_csv("rf_benchmark.csv", index=False)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
