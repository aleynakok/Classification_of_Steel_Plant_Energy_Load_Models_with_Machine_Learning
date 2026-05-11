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

for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=col, kde=True, bins=50)
    #plt.title(f'{col} s Histogram', fontsize=15)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.savefig(f"histogram{col}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

data['date'] = pd.to_datetime(data['date'], format='%d/%m/%Y %H:%M')
data_daily_avg = data.set_index('date')['Usage_kWh'].resample('D').mean()
plt.figure(figsize=(16, 7))
data_daily_avg.plot()
#plt.title('Daily Average Energy Consumption (Usage_kWh)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Usage_kWh', fontsize=12)
plt.savefig(f"lineplot{col}.pdf", format="pdf", bbox_inches="tight")
plt.show()

data_sample = data.sample(n=2000, random_state=42)
plt.figure(figsize=(12, 7))
sns.scatterplot(data=data_sample, x='CO2(tCO2)', y='Usage_kWh', hue='Load_Type', alpha=0.7)
#plt.title('Usage_kWh vs CO2(tCO2) Relationship', fontsize=15)
plt.xlabel('CO2(tCO2)', fontsize=12)
plt.ylabel('Usage_kWh', fontsize=12)
plt.savefig("scatter1.pdf", format="pdf", bbox_inches="tight")
plt.show()

plt.figure(figsize=(12, 7))
sns.scatterplot(data=data_sample, x='Lagging_Current_Reactive.Power_kVarh', y='Usage_kWh', hue='Load_Type', alpha=0.7)
#plt.title('Usage_kWh vs Lagging...Power_kVarh Relationship', fontsize=15)
plt.xlabel('Lagging Current Reactive Power (kVarh)', fontsize=12)
plt.ylabel('Usage_kWh', fontsize=12)
plt.savefig("scatter2.pdf", format="pdf", bbox_inches="tight")
plt.show()

corr_matrix = data[numeric_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            linewidths=0.5)
#plt.title('Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("heatmap.pdf", format="pdf", bbox_inches="tight")
plt.show()