import warnings
# Sadece jupyter_client modülünden gelen uyarıları gizledik
warnings.filterwarnings("ignore", module="jupyter_client")

# ASSOCIATION MINING – APRIORI ALGORITHM

from mlxtend.frequent_patterns import apriori, association_rules

# 1. DISCRETIZATION (Low–Med–High)
df['Usage_bin'] = pd.cut(df['Usage_kWh'], [-1,15,40,df['Usage_kWh'].max()],
                         labels=['LowUsage','MedUsage','HighUsage'])

df['LagReactive_bin'] = pd.cut(df['Lagging_Current_Reactive.Power_kVarh'], [-1,8,25, df['Lagging_Current_Reactive.Power_kVarh'].max()],
                               labels=['LowReac','MedReac','HighReac'])

df['LagPF_bin'] = pd.cut(df['Lagging_Current_Power_Factor'], [-1,60,85, df['Lagging_Current_Power_Factor'].max()],
                         labels=['LowPF','MedPF','HighPF'])

df['Time_bin'] = pd.cut(df['NSM'], [-1,21600,43200,64800,86400],
                        labels=['Night','Morning','Noon','Evening'])

df['Load_Type'] = df['Load_Type'].str.replace(" ","")

# 2. ONE-HOT ENCODING FOR APRIORI
cols = ['Usage_bin','LagReactive_bin','LagPF_bin','Time_bin','Day_of_week','Load_Type']
# her kategori için bir sütun açtık, mesela "Time_bin" gider yerine "Time_bin_Night"gelir.
df_encoded = pd.get_dummies(df[cols])

# 3. APRIORI
frequent_items = apriori(df_encoded, min_support=0.05, use_colnames=True)

print("\nTOP FREQUENT ITEMSETS:")
print(frequent_items.sort_values("support", ascending=False).head(10))

# 4. ASSOCIATION RULES
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.6)
# lift > 1 ise positive correlation
rules = rules.sort_values("lift", ascending=False)

print("\nTOP RULES BY LIFT:")
print(rules[['antecedents','consequents','support','confidence','lift']].head(10))


# ENSEMBLE MODELS – AdaBoost & RandomForest

import warnings
warnings.filterwarnings("ignore")

# GEREKSİZ SÜTUNLARI ATMA
if 'date' in df.columns:
    df_model = df.drop(columns=['date'])

# Hedef ve Özellikler
y = df_model['Load_Type']
X = df_model.drop(columns=['Load_Type'])

# Kategorik verileri sayısal hale getirme (One-Hot Encoding)
# drop_first=True diyerek dummy trap'ten kurtuluruz ve sütun sayısını azaltırız.("dummy variable trap" hatasını önler)
X = pd.get_dummies(X, drop_first=True)

# Hedef değişkeni (Light, Medium, Maximum_Load) sayısal yapalım
le = LabelEncoder()
y = le.fit_transform(y)

# 2. SPLIT + SCALE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# 3. ADABOOST
# Not: Sklearn yeni sürümlerinde base_estimator yerine 'estimator' parametresi kullanılır.
# Uyarı alırsan base_estimator yerine estimator yazabilirsin.
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2), # basit ağaç, max_depth=2
    n_estimators=120, #120 tane desicion tree kur ve hataları çözerek ilerle
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train_s, y_train)
ada_pred = ada.predict(X_test_s)

print("\nADABOOST RESULTS")
# LabelEncoder kullandığımız için isimleri geri getirelim raporlarken
print("Accuracy:", accuracy_score(y_test, ada_pred))
print(classification_report(y_test, ada_pred, target_names=le.classes_))

# 4. RANDOM FOREST
# 200 tane ağaç birbirinden bağımsız tahmin yapar.
# voting yapılır, majority vote a bakılır
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train_s, y_train)
rf_pred = rf.predict(X_test_s)

print("\nRANDOM FOREST RESULTS")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred, target_names=le.classes_))