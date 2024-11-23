import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Data Preprocessing For Decision Tree
'''
df = pd.read_csv("C:/Users/User/Downloads/insurance.csv")
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encoding for 'region'
# Perform One-Hot Encoding on 'region' column
df_encoded = pd.get_dummies(df, columns=['region'], drop_first=False) 
# Convert True/False to 1/0 for the region columns
df_encoded['region_northeast'] = df_encoded['region_northeast'].astype(int)
df_encoded['region_northwest'] = df_encoded['region_northwest'].astype(int)
df_encoded['region_southeast'] = df_encoded['region_southeast'].astype(int)
df_encoded['region_southwest'] = df_encoded['region_southwest'].astype(int)

df_encoded.to_csv('C:/Users/User/Downloads/insurance_decision.csv', index=False)
'''

df = pd.read_csv("C:/Users/User/Downloads/insurance_decision.csv")
# Define features (X) and target (y)
X = df.drop(columns=['charges'])  # Assuming 'charges' is the target variable
y = df['charges']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree regression model
dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# Define the hyperparameters to test
param_grid = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 10, 20],
    'min_samples_leaf': [1, 5, 10]
}

# Perform GridSearchCV
grid_search = GridSearchCV(dt_model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters and evaluation
print(f"Best Parameters: {grid_search.best_params_}")
best_dt_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = dt_model.predict(X_test)

# Predict using the best model
y_pred_dt = best_dt_model.predict(X_test)

# Access feature importance
importance = best_dt_model.feature_importances_

# Display the importance for each feature
for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Score: {v:.5f}')

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print evaluation metrics
print(f"Test Score: {dt_model.score(X_test, y_test)}")
print(f"Training Score: {dt_model.score(X_train, y_train)}")
print(f"Decision Tree Regression - Mean Squared Error: {mse}")
print(f"Decision Tree Regression - R-squared: {r2}")
print(f"Decision Tree (Best Model) - Mean Squared Error: {mse_dt}")
print(f"Decision Tree (Best Model) - R-squared: {r2_dt}")

# Graph for comparison
plt.figure(figsize=(12, 6))

# Scatterplot for base model
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Base Decision Tree Regression: Actual vs Predicted')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.grid()

# Scatterplot for best model
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_dt, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Best Decision Tree Regression: Actual vs Predicted')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.grid()

plt.tight_layout()
plt.show()
