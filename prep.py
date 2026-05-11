col_to_check = 'Usage_kWh'
plt.figure(figsize=(12, 3))
sns.boxplot(data=data, x=col_to_check)
#plt.title(f'{col_to_check} - Before Box Plot')
plt.savefig(f"beforeafter1.0.pdf", format="pdf", bbox_inches="tight")
plt.show()

Q1 = data[col_to_check].quantile(0.25)
Q3 = data[col_to_check].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
print(f"Lower Bound: {lower_bound:.2f}")
print(f"Upper Bound: {upper_bound:.2f}")

outliers_lower = data[data[col_to_check] < lower_bound]
outliers_upper = data[data[col_to_check] > upper_bound]
total_outliers = len(outliers_lower) + len(outliers_upper)
print(f"{total_outliers} many outliers detected.")

col_clipped = col_to_check + '_clipped'
data[col_clipped] = data[col_to_check].clip(lower=lower_bound, upper=upper_bound)


plt.figure(figsize=(12, 3))
sns.boxplot(data=data, x=col_clipped)
#plt.title(f'{col_clipped} - After Box Plot')
plt.savefig(f"beforeafter1.1.pdf", format="pdf", bbox_inches="tight")
plt.show()

bins = [0, 21600, 43200, 64800, 86401]
labels = ['Night', 'Morning', 'Noon', 'Evening']
data['GununZamani'] = pd.cut(data['NSM'], bins=bins, labels=labels, right=True)
print(data['GununZamani'].value_counts().sort_index())
if 'GununZamani' not in categorical_cols:
    categorical_cols.append('GununZamani')

col_std = 'Usage_kWh_clipped'
col_norm = 'Lagging_Current_Power_Factor'

scaler_std = StandardScaler()
col_std_yeni = col_std + '_standardized'
data[col_std_yeni] = scaler_std.fit_transform(data[[col_std]])
print(f"'{col_std_yeni}' column added (Mean=0, Std=1).")
scaler_norm = MinMaxScaler()

col_norm_yeni = col_norm + '_normalized'
data[col_norm_yeni] = scaler_norm.fit_transform(data[[col_norm]])
print(f"'{col_norm_yeni}' column added ( 0-1).")
print(data[[col_std, col_std_yeni, col_norm, col_norm_yeni]].head())
print(data[[col_std_yeni, col_norm_yeni]].describe())

col_to_encode = 'Day_of_week'
one_hot_encoded = pd.get_dummies(data[col_to_encode], prefix=col_to_encode)
data = pd.concat([data, one_hot_encoded], axis=1)
new_cols = [col_to_encode] + list(one_hot_encoded.columns[:3])
print(data[new_cols].head())

numeric_cols_for_pca = [
    'Usage_kWh', 'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh', 'CO2(tCO2)',
    'Lagging_Current_Power_Factor', 'Leading_Current_Power_Factor', 'NSM'
]
data_numeric = data[numeric_cols_for_pca]
scaler_pca = StandardScaler()
data_scaled = scaler_pca.fit_transform(data_numeric)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data_scaled)
pca_data = pd.DataFrame(data = principal_components,
                      columns = ['PC1', 'PC2'])
pca_data = pd.concat([pca_data, data[['Load_Type']]], axis=1)
plt.figure(figsize=(12, 8))
sns.scatterplot(data=pca_data,
                x='PC1',
                y='PC2',
                hue='Load_Type',
                alpha=0.6)
#plt.title('2D VISUALIZATION (Based on Load_Type\)', fontsize=16)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.legend(title='Load Type')
plt.grid(True)
plt.savefig(f"PCA2D.pdf", format="pdf", bbox_inches="tight")
plt.show()
print(f"\nexplained variance by PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"explained variance by PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"total explained variance (PC1 + PC2): {pca.explained_variance_ratio_.sum():.2%}")

scaler_pca_3d = StandardScaler()
data_scaled_3d = scaler_pca_3d.fit_transform(data_numeric)
pca_3d = PCA(n_components=3)
principal_components_3d = pca_3d.fit_transform(data_scaled_3d)
pca_data_3d = pd.DataFrame(data = principal_components_3d,
                         columns = ['PC1', 'PC2', 'PC3'])


pca_data_3d = pd.concat([pca_data_3d, data[['Load_Type']]], axis=1)
data_sample_3d = pca_data_3d.sample(n=2000, random_state=42)
fig = plt.figure(figsize=(13, 10))
ax = fig.add_subplot(111, projection='3d')
targets = data_sample_3d['Load_Type'].unique()
colors_map = {'Light_Load': 'green', 'Medium_Load': 'blue', 'Maximum_Load': 'red'}
for target in targets:
    indices_to_keep = data_sample_3d['Load_Type'] == target
    ax.scatter(data_sample_3d.loc[indices_to_keep, 'PC1'],
               data_sample_3d.loc[indices_to_keep, 'PC2'],
               data_sample_3d.loc[indices_to_keep, 'PC3'],
               c = colors_map[target],
               s = 40,
               alpha = 0.6,
               label = target)
ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_zlabel('PC3', fontsize=12)
ax.legend(title='Load Type')
ax.grid(True)
plt.savefig(f"PCA3D.pdf", format="pdf", bbox_inches="tight")
plt.show()
print(f"explained variance by PC1: {pca_3d.explained_variance_ratio_[0]:.2%}")
print(f"explained variance by PC2 : {pca_3d.explained_variance_ratio_[1]:.2%}")
print(f"explained variance by PC3 : {pca_3d.explained_variance_ratio_[2]:.2%}")
print("-" * 30)
print(f"total explained variance (PC1 + PC2 + PC3): {pca_3d.explained_variance_ratio_.sum():.2%}")

data_model = data.copy()
cols_to_drop = [col for col in data_model.columns if
                '_standardized' in col or
                '_normalized' in col or
                '_clipped' in col or
                'Day_of_week_' in col or
                'WeekStatus_Weekend' in col or
                'date' in col or
                'GununZamani' in col]
data_model = data_model.drop(columns=cols_to_drop)
categorical_features = ['WeekStatus', 'Day_of_week', 'Load_Type']
encoder = OrdinalEncoder()
data_model[categorical_features] = encoder.fit_transform(data_model[categorical_features])
print(data_model[['WeekStatus', 'Day_of_week', 'Load_Type']].head())
TARGET_VARIABLE = 'Load_Type'
y = data_model[TARGET_VARIABLE]
X = data_model.drop(columns=[TARGET_VARIABLE])
print(f"\nTarget variable (y): {TARGET_VARIABLE}")
print(f"features (X): {list(X.columns)}")
model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X, y)
importances = model.feature_importances_
feature_names = X.columns
importance_data = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(importance_data)
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_data, x='Importance', y='Feature')
#plt.title('Random Forest Feature Importance', fontsize=16)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.savefig(f"feature_importance.pdf", format="pdf", bbox_inches="tight")
plt.show()