df_reg = data.copy()
df_reg = pd.get_dummies(df_reg, columns=['WeekStatus', 'Day_of_week'], drop_first=True)

# Modelin geçmiş tüketim değerlerinden öğrenmesi tahmin gücünü artırır, ondan ekliyorum.
# 'shift()' fonksiyonu ile geçmiş zaman adımlarını yeni sütunlar olarak ekliyorum.
df_reg['Usage_kWh_lag1'] = df_reg['Usage_kWh'].shift(1) # 15 dakika önceki tüketim
df_reg['Usage_kWh_lag4'] = df_reg['Usage_kWh'].shift(4) # 1 saat önceki tüketim

columns_to_drop_reg = ['date', 'Load_Type', 'CO2(tCO2)']
df_reg = df_reg.drop(columns=columns_to_drop_reg)

# shift() fonksiyonu nedeniyle oluşan NaN değerli ilk satırları temizleyedim.
df_reg = df_reg.dropna()
print("Regresyon için veri seti hazırlandı.")

X_reg = df_reg.drop('Usage_kWh', axis=1)
y_reg = df_reg['Usage_kWh']
print(f"Target Attribute (y): Usage_kWh")
print(f"Attributelar (X): {list(X_reg.columns)}")

# Train ve Test
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)
print(f"\nEğitim seti boyutu: {X_train_reg.shape}")
print(f"Test seti boyutu: {X_test_reg.shape}")

models_reg = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
}

results_reg = {}
predictions_reg = {}

for name, model in models_reg.items():
    print(f"\n--- {name} Modeli İşleniyor ---")

    model.fit(X_train_reg, y_train_reg)

    y_pred_reg = model.predict(X_test_reg)
    predictions_reg[name] = y_pred_reg

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)
    results_reg[name] = {'RMSE': rmse, 'MAPE': mape}
    print(f"Performans Sonuçları: -> RMSE: {rmse:.4f}, MAPE: {mape:.4%}")

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_test_reg, y=y_pred_reg, alpha=0.5, s=20)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], '--r', linewidth=2, label='İdeal Çizgi (Y=X)')
    plt.title(f"{name}: Real vs. Predicted values", fontsize=16)
    plt.xlabel("Real values (Usage_kWh)")
    plt.ylabel("Predicted values (Usage_kWh)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{name.replace(' ', '_')}_actual_vs_predicted.pdf", format="pdf", bbox_inches="tight")
    plt.show()

rf_model_reg = models_reg["Random Forest"]

importances_reg = rf_model_reg.feature_importances_
feature_names_reg = X_reg.columns
rf_importance_df_reg = pd.DataFrame({'Feature': feature_names_reg, 'Importance': importances_reg}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(data=rf_importance_df_reg.head(10), x='Importance', y='Feature', hue='Feature', palette='viridis', legend=False)
plt.title('Random Forest - The most important 10 attributes', fontsize=16)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Attribute', fontsize=12)
plt.savefig("rf_regressor_feature_importance.pdf", format="pdf", bbox_inches="tight")
plt.show()

results_df_reg = pd.DataFrame(results_reg).T
print("\n--- Summary of Model Performances ---")
print(results_df_reg.sort_values(by='RMSE'))