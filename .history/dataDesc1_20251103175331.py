import os, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

data=pd.read_csv("Steel_industry_data.csv")
data

print(data.shape)
data.info()
print(data.isnull().sum())
print(data.head())

numeric_cols = [
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
    'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM'
]

categorical_cols = [
    'WeekStatus', 'Day_of_week', 'Load_Type']
for col in categorical_cols:
    print(f"'{col}' Column's Mode: {data[col].mode()[0]}")


for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    order = data[col].value_counts().index
    sns.countplot(data=data, x=col, order=order)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.savefig(f"barplots{col}.pdf", format="pdf", bbox_inches="tight")


for col in numeric_cols:
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=data, x=col)
    #plt.title(f'{col} s Box-plot', fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.savefig(f"boxplots{col}.pdf", format="pdf", bbox_inches="tight")
    plt.show()