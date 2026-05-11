df_model_cls = data.copy()

columns_to_drop_cls = ['date', 'CO2(tCO2)', 'WeekStatus']
df_model_cls = df_model_cls.drop(columns=columns_to_drop_cls)
print(f"Sınıflandırma için '{', '.join(columns_to_drop_cls)}' sütunları droplandı.")

le = LabelEncoder()
df_model_cls['Load_Type'] = le.fit_transform(df_model_cls['Load_Type'])
load_type_classes = le.classes_

# Girdi özniteliklerindeki (X) kategorik sütunu One-Hot Encoding ile dönüştürelim
df_model_cls = pd.get_dummies(df_model_cls, columns=['Day_of_week'], drop_first=True)

print("Sınıflandırma için veri seti modellemeye hazır.")
print(df_model_cls.head())

print("\n--- CLASSIFICATION TASK ---")

X_cls = df_model_cls.drop('Load_Type', axis=1)
y_cls = df_model_cls['Load_Type']

# Train and Test
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.3, random_state=42, stratify=y_cls
)
print(f"\nEğitim seti boyutu: {X_train_cls.shape}")
print(f"Test seti boyutu: {X_test_cls.shape}")

# Training
models_cls = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
}

# Sonuçları saklamak için boş bir dictionary
results_cls = {}

#accuracy
for name, model in models_cls.items():
    print(f"\n--- {name} Modeli İşleniyor ---")
    model.fit(X_train_cls, y_train_cls)
    y_pred_cls = model.predict(X_test_cls)

    accuracy = accuracy_score(y_test_cls, y_pred_cls)
    results_cls[name] = accuracy

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls, y_pred_cls, target_names=load_type_classes))

print("\n--- Karar Ağacı Görselleştirmesi ---")
dt_model_viz = models_cls["Decision Tree"]
plt.figure(figsize=(35, 15))
plot_tree(dt_model_viz,
          feature_names=X_cls.columns,
          class_names=load_type_classes,
          filled=True, rounded=True, fontsize=10, impurity=False)
plt.title("Decision Tree Visualization", fontsize=20)
plt.savefig("decision_tree_visualization.pdf", format="pdf", bbox_inches="tight")
plt.show()

print("\n--- Random Forest Grafiği ---")
rf_model_viz = models_cls["Random Forest"]
importances_cls = rf_model_viz.feature_importances_
feature_names_cls = X_cls.columns
rf_importance_df_cls = pd.DataFrame({'Feature': feature_names_cls, 'Importance': importances_cls}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=rf_importance_df_cls.head(10), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Random Forest - The most important 10 attributes', fontsize=16)
plt.savefig("rf_classifier_feature_importance.pdf", format="pdf", bbox_inches="tight")
plt.show()

print("\n--- Lojistik Regresyon Grafiği ---")
log_reg_model = models_cls["Logistic Regression"].named_steps['logisticregression']
coefficients = log_reg_model.coef_

for i, class_name in enumerate(load_type_classes):
    lr_coeffs_df = pd.DataFrame({
        'Feature': X_cls.columns,
        'Coefficient': coefficients[i]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=lr_coeffs_df.head(10), x='Coefficient', y='Feature', hue='Feature', palette='coolwarm', legend=False)
    plt.title(f'Logistic Regression: The most important 10 attributes (Class: {class_name})', fontsize=14)
    plt.xlabel('Coefficient', fontsize=12)
    plt.ylabel('Attribute', fontsize=12)
    plt.savefig(f"logistic_regression_coeffs_{class_name}.pdf", format="pdf", bbox_inches="tight")
    plt.show()

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)