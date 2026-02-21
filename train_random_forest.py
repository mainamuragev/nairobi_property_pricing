# train_random_forest.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('cleaned_properties.csv')
print("Columns:", df.columns.tolist())

# Features (same as baseline)
target = 'price_normalized'
numeric_features = ['bedrooms']
categorical_features = ['location']

# Prepare data
df_model = df.dropna(subset=[target] + numeric_features + categorical_features)
X = df_model[numeric_features + categorical_features]
y = df_model[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# --- Linear Regression (baseline) ---
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)

lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
lr_r2 = r2_score(y_test, y_pred_lr)

# --- Random Forest ---
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
rf_r2 = r2_score(y_test, y_pred_rf)

# --- Model Comparison ---
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"{'Model':<20} {'MAE (KES)':>15} {'RMSE (KES)':>15} {'R²':>10}")
print("-"*60)
print(f"{'Linear Regression':<20} {lr_mae:>15,.0f} {lr_rmse:>15,.0f} {lr_r2:>10.3f}")
print(f"{'Random Forest':<20} {rf_mae:>15,.0f} {rf_rmse:>15,.0f} {rf_r2:>10.3f}")
print("="*60)

# Improvement
mae_improvement = (lr_mae - rf_mae) / lr_mae * 100
print(f"\nRandom Forest improved MAE by {mae_improvement:.1f}%")

# --- Feature Importance ---
# Get feature names after one-hot encoding
feature_names = (numeric_features +
                 list(rf_pipeline.named_steps['preprocessor']
                      .named_transformers_['cat']
                      .get_feature_names_out(categorical_features)))

importances = rf_pipeline.named_steps['regressor'].feature_importances_

# Create DataFrame for plotting
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

# Show top 10 for context, but we'll plot top 5
print("\nTop 10 Feature Importances:")
print(feat_imp.head(10).to_string(index=False))

# Plot top 5 and save to file (instead of showing)
plt.figure(figsize=(10, 6))
top5 = feat_imp.head(5)

# Fixed seaborn warning by adding hue and legend=False
sns.barplot(data=top5, x='importance', y='feature', 
            hue='feature', palette='viridis', legend=False)

plt.title('Top 5 Price Drivers')
plt.xlabel('Importance')
plt.tight_layout()

# Save the plot
plot_filename = 'feature_importance_top5.png'
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.close()
print(f"✅ Feature importance plot saved as '{plot_filename}'")

# --- Save the best model (Random Forest) ---
joblib.dump(rf_pipeline, 'model.pkl')
print("✅ Best model saved as 'model.pkl'")
