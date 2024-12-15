#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:20:37 2024

@author: zhongjingxin
"""

# 第二节：生存分析与可视化——以两种治疗方案治疗肿瘤患者的生存时间分析为例

# 在Python中，可以利用lifelines包进行生存分析。其方法包括：
# ①通过Kaplan-Meier plots绘图使生存曲线可视化；
# ②通过Nelson-Aalen plots绘图可视化累积危害风险；
# ③通过Log-Rank test检验比较两组或更多组的生存曲线；
# ④通过Cox比例风险回归，揭示不同变量对生存的影响。


# 一、实践数据

# A、B两种治疗方案（简称A组、B组）分别治疗某恶性肿瘤患者25人和22人，随访记录患者的生存时间（月）如下，“+”表示删失数据（无法得知随访对象的确切生存时间者）。


# 二、实践任务

# （1）分析生存率和绘制生存曲线。
# （2）分析死亡风险和绘制累积死亡概率曲线。
# （3）对两组生存率进行比较。
# （4）对两组的累积风险进行比较。


# 三、实践程序及实践结果

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test  # 确保正确导入 logrank_test
from lifelines.utils import median_survival_times

# 数据加载与预处理
data = pd.DataFrame({
    'Group': ['A组']*25 + ['B组']*22,
    'Time': [10, 2, 12, 13, 18, 6, 19, 26, 9, 8, 6, 43, 9, 4, 31, 24, 23, 20, 18, 13, 15, 30, 28, 35, 37] + 
            [2, 13, 7, 11, 6, 1, 11, 3, 17, 7, 22, 33, 22, 20, 10, 9, 21, 16, 19, 25, 19, 17],
    'Status': [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1] + 
               [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
})

### 第一部分：分析生存率和绘制生存曲线
def analyze_survival_rate(data, group):
    kmf = KaplanMeierFitter()
    group_data = data[data['Group'] == group]
    kmf.fit(group_data['Time'], group_data['Status'], label=group)
    
    print(f'\n{group} Median Survival Time and 95% CI:\n', 
          kmf.median_survival_time_, '\n',
          median_survival_times(kmf.confidence_interval_))
    
    return kmf.plot(show_censors=True, ci_show=False)

fig, ax = plt.subplots(figsize=(10, 6))
for group in ['A组', 'B组']:
    analyze_survival_rate(data, group)

ax.set_xlabel('时间（月）/ Time (months)', fontsize=12)
ax.set_ylabel('生存率 / Survival Rate', fontsize=12)
ax.set_title("A组和B组的生存率曲线对比图 / Comparison of Survival Curves for Group A and B", y=-0.26, fontsize=15)
plt.show()

### 第二部分：分析死亡风险和绘制累积死亡概率曲线
def plot_cumulative_death_probability(data, group, ax):
    kmf = KaplanMeierFitter()
    group_data = data[data['Group'] == group]
    kmf.fit(group_data['Time'], group_data['Status'], label=group)
    print(f'\n{group} Cumulative Death Probability:\n', kmf.cumulative_density_)
    kmf.plot_cumulative_density(ax=ax, label=f'{group}')

fig, ax = plt.subplots(figsize=(10, 6))
for group in ['A组', 'B组']:
    plot_cumulative_death_probability(data, group, ax)

ax.set_xlabel('时间（月）/ Time (months)', fontsize=12)
ax.set_ylabel('累积死亡概率 / Cumulative Death Probability', fontsize=12)
ax.set_title('A组和B组的累积死亡概率对比图 / Comparison of Cumulative Death Probabilities for Group A and B', y=-0.25, fontsize=15)
plt.legend(loc='upper left')
plt.show()

### 第三部分：对两组生存率进行比较
lr_test = logrank_test(
    data['Time'][data['Group'] == 'A组'],
    data['Time'][data['Group'] == 'B组'],
    event_observed_A=data['Status'][data['Group'] == 'A组'],
    event_observed_B=data['Status'][data['Group'] == 'B组'],
    alpha=0.95
)
print('\nLog-Rank Test Results for Group A and B:\n', lr_test.summary)

### 第四部分：对两组的累积风险进行比较
def compare_cumulative_risk(data, groups, axes):
    for i, group in enumerate(groups):
        group_data = data[data['Group'] == group]
        
        # Kaplan-Meier analysis
        kmf = KaplanMeierFitter()
        kmf.fit(group_data['Time'], group_data['Status'])
        print(f'\n{group} Event Table (KM):\n', kmf.event_table)
        print(f'{group} Cumulative Density (KM):\n', kmf.cumulative_density_)
        kmf.plot_cumulative_density(label=f'{group} cumulative_density', ax=axes[i])
        
        # Nelson-Aalen analysis
        naf = NelsonAalenFitter()
        naf.fit(group_data['Time'], group_data['Status'])
        print(f'{group} Event Table (NA):\n', naf.event_table)
        print(f'{group} Cumulative Hazard (NA):\n', naf.cumulative_hazard_)
        print(f'{group} 95% CI of Cumulative Hazard (NA):\n', naf.confidence_interval_cumulative_hazard_)
        naf.plot_cumulative_hazard(label=f'{group} cumulative_hazard', ax=axes[i])
        
        # Set graph properties
        axes[i].set_xlabel('时间（月）/ Time (months)', fontsize=12)
        axes[i].set_ylabel('累积风险密度/累积风险 / Cumulative Hazard Density/Cumulative Hazard', fontsize=12)
        axes[i].legend(loc='upper left')
        axes[i].set_title(f'累积死亡风险概率密度与累积风险 / Cumulative Death Risk Probability Density and Cumulative Hazard for Group {group}', y=-0.25, fontsize=14)

fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)
compare_cumulative_risk(data, ['A组', 'B组'], axes)
plt.tight_layout()
plt.show()

# 绘制两组的累积风险概率密度和累积风险比较图
fig, axes = plt.subplots(1, 2, figsize=(20, 6))

# 累积风险概率密度
for group in ['A组', 'B组']:
    group_data = data[data['Group'] == group]
    kmf = KaplanMeierFitter()
    kmf.fit(group_data['Time'], group_data['Status'])
    kmf.plot_cumulative_density(label=group, ax=axes[0])

axes[0].set_xlabel('时间（月）/ Time (months)', fontsize=12)
axes[0].set_ylabel('累积风险密度 / Cumulative Hazard Density', fontsize=12)
axes[0].set_title('A组和B组的累积死亡风险概率密度比较图 / Comparison of Cumulative Death Risk Probability Density for Group A and B', y=-0.25, fontsize=14)
axes[0].legend(loc='lower right')

# 累积风险
for group in ['A组', 'B组']:
    group_data = data[data['Group'] == group]
    naf = NelsonAalenFitter()
    naf.fit(group_data['Time'], group_data['Status'])
    naf.plot_cumulative_hazard(label=group, ax=axes[1])

axes[1].set_xlabel('时间（月）/ Time (months)', fontsize=12)
axes[1].set_ylabel('累积风险 / Cumulative Hazard', fontsize=12)
axes[1].set_title('A组和B组的累积死亡风险比较图 / Comparison of Cumulative Death Risk for Group A and B', y=-0.25, fontsize=14)
axes[1].legend(loc='upper left')

plt.tight_layout()
plt.show()


