import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('cleaned_properties.csv')
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows of key columns:")
print(df[['price', 'price_normalized', 'location', 'bedrooms']].head())

# --- Use the already normalized price column ---
target = 'price_normalized'           # this is the cleaned numeric price
numeric_features = ['bedrooms']        # add other numeric if available (e.g., 'bathrooms')
categorical_features = ['location']    # location is categorical

# Check that selected columns exist
available_cols = df.columns.tolist()
for col in [target] + numeric_features + categorical_features:
    if col not in available_cols:
        print(f"❌ ERROR: '{col}' not found. Please adjust feature lists.")
        exit()

# Drop rows with missing values in these columns
df_model = df.dropna(subset=[target] + numeric_features + categorical_features)
print(f"\nRows after dropping missing: {len(df_model)} out of {len(df)}")

X = df_model[numeric_features + categorical_features]
y = df_model[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing: one-hot encode categoricals
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("BASELINE LINEAR REGRESSION PERFORMANCE")
print("="*50)
print(f"MAE  = KES {mae:,.0f}")
print(f"RMSE = KES {rmse:,.0f}")
print(f"R²   = {r2:.3f}")
print("="*50)
print(f"\nInterpretation: On average, predictions are off by KES {mae:,.0f}.")
