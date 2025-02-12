import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
file_path = 'BTC1min.csv' 
data = pd.read_csv(file_path)
print("Data loaded successfully.")

# Check for null values in the dataset
print("Checking for null values...")
null_values = data.isnull().sum()
print("Null values in each column:\n", null_values)

# Drop rows with null values only in the 'Close' column
data = data[data['Close'].notna()]
print(f"Dropped rows with NA values in 'Close'. Remaining rows: {data.shape[0]}.")

# Display the first few rows of the cleaned dataset
print("First few rows of the cleaned dataset:\n", data.head())

# Prepare features and target variable
X = data.drop(columns=['Close'])  # Features (all columns except 'Close')
y = data['Close']  # Target variable (closing price)
print("Features and target variable prepared.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

# Initialize the XGBoost Regressor with GPU support
model = XGBRegressor(n_estimators=100, tree_method='gpu_hist', gpu_id=0, random_state=42)
print("XGBoost model initialized with GPU support.")

# Train the model
model.fit(X_train, y_train)
print("Model training completed.")

# Make predictions on the test set
y_pred = model.predict(X_test)
print("Predictions made on the test set.")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Visualize predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Closing Prices')
plt.ylabel('Predicted Closing Prices')
plt.title('Actual vs Predicted Closing Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)  # Diagonal line
plt.show()
print("Visualization of predictions vs actual values completed.")

# Optionally, visualize residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
print("Histogram of residuals displayed.")

# Feature Importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {X.columns[indices[f]]} ({importances[indices[f]]})")

# Optionally, plot the feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
print("Feature importances plotted.")

# Save the model to a file
import joblib
joblib.dump(model, 'xgboost_model.pkl')
print("Model saved successfully.")