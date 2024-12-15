#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 15:03:15 2024

@author: dx
"""

import sys
import os

# 获取主程序的第一个参数，即脚本名称
script_name = sys.argv[0]

# 如果是通过命令行运行，则可以获得绝对路径；否则可能是相对路径
script_path = os.path.abspath(script_name)
print("当前脚本的绝对路径:", script_path)

# 获取当前脚本所在的目录
script_dir = os.path.dirname(script_path)
print("当前脚本所在目录:", script_dir)

# 更改工作目录为脚本所在的目录
os.chdir(script_path)
print("当前工作目录已更改为:", os.getcwd())

# 第一节：Logistic回归分析——以成年人体质分类的影响因素分析为例

# （一）实践数据
# 数据文件为“/PyData2403/GdAdultPhy1000_04.xlsx”。   

 
# （二）实践任务
# 以体质分类（二分类）为因变量，以性别、身高和体重为自变量拟合Logistic回归。


# （三）实践程序及说明


import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

# 数据加载与预处理
file_path = './PyData2403/GdAdultPhy1000_04.xlsx'
dataFram = pd.read_excel(file_path, index_col='Number', engine='openpyxl')
Sex_dummy = pd.get_dummies(dataFram['Sex'], prefix='Sex', dtype=int)
dataFram_encoded = pd.concat([dataFram, Sex_dummy], axis=1)

# 定义因变量和自变量
Y = dataFram['PhysiLv2']
X = dataFram_encoded[['Sex_2', 'Height', 'Weight']]

# 使用statsmodels进行Logistic回归分析
X1 = sm.add_constant(X)  # 添加截距项
logit_model = sm.Logit(Y, X1).fit()
print('1-1 Logit模型拟合结果：\n', logit_model.summary())
predicts = logit_model.predict(X1)
print('1-2 Logit模型预测值(前5个)：\n', predicts[:5])

# 使用sklearn进行Logistic回归分析
LogiReg_model = LogisticRegression(penalty="l2", fit_intercept=True)
LogiReg_model.fit(X, Y)

# 打印sklearn模型信息
print('2-1 模型采用的参数：\n', LogiReg_model.get_params())
print('2-2 拟合分类标签：', LogiReg_model.classes_)
print('2-3 拟合特征数：', LogiReg_model.n_features_in_)
print('2-4 拟合特征名：', LogiReg_model.feature_names_in_)
print('2-5 模型常数项Intercept：', LogiReg_model.intercept_)
print('2-6 自变量系数Coef：', LogiReg_model.coef_)
print('2-7 模型预测的准确率Score：', LogiReg_model.score(X, Y))

# 预测与概率预测
predicts_sklearn = LogiReg_model.predict(X)
predict_probas = LogiReg_model.predict_proba(X)

print('2-8 Sklearn预测结果（前6个）：\n', predicts_sklearn[:6])
print('2-9 Sklearn预测结果概率值（前6个）：\n', predict_probas[:6])

