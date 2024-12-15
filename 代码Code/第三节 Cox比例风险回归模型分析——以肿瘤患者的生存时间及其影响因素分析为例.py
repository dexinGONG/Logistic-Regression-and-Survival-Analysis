#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:20:59 2024

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
os.chdir(script_dir)
print("当前工作目录已更改为:", os.getcwd())



# 第三节：Cox比例风险回归模型分析——以肿瘤患者的生存时间及其影响因素分析为例

# 一、实践数据

# A、B两种治疗方案（简称A组、B组）分别治疗某恶性肿瘤患者25人和22人，患者基本情况包括性别
# （男性=1，女性=2）、年龄（岁）和体重（kg），病情包括严重程度分级（分为1、2、3级）、
# 是否转移（转移=1，未转移=0）。随访记录患者的生存时间（月）和结局（死亡=1，删失=0）。


# 二、实践任务

# 以性别、年龄、疗法等因素为协变量，采用Cox回归模型分析患者的生存时间及状态的影响因素。

# 三、实践程序及说明

# Model 5: Cox Proportional Hazards Regression Analysis for Group A and B

import pandas as pd
from lifelines import CoxPHFitter

# 一、数据准备
# 加载数据集并进行初步处理

data = pd.read_excel("./PyData2403/AB组COX生存分析数据.xlsx")

# 删除不纳入分析的列数据
data.drop('编号', axis=1, inplace=True)

# 将疗法分类转换为数值型变量（A组 -> 1, B组 -> 2）
data['疗法'] = data['疗法'].map({'A组': 1, 'B组': 2})

# 二、初步Cox回归分析
# 创建Cox比例风险模型实例
cph_initial = CoxPHFitter()

# 拟合初步模型
cph_initial.fit(data, duration_col='生存时间', event_col='结局')

# 输出初步Cox回归预分析结果
print('\n（1）Cox Regression Preliminary Analysis Results (初步Cox回归预分析结果)\n')
cph_initial.print_summary()

# 三、根据预分析结果选取纳入分析的数据并进行最终Cox回归分析
# 根据预分析结果选择特征
selected_columns = ['疗法', '是否转移', '体重', '分级', '生存时间', '结局']
data_final = data[selected_columns]

# 创建新的Cox比例风险模型实例
cph_final = CoxPHFitter()

# 拟合最终模型
cph_final.fit(data_final, duration_col='生存时间', event_col='结局')

# 输出最终Cox回归分析结果
print('\n（2）Final Cox Regression Analysis Results for Factors Influencing Survival Time in Patients with Certain Malignant Tumors (某种恶性肿瘤患者生存时间影响因素的Cox回归分析结果)\n')
cph_final.print_summary()




