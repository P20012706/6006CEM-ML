import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("C:/Users/User/Downloads/insurance_linear.csv")

# Create the health_status feature
df['health_status'] = df['age'] * df['bmi'] * df['smoker']

# Define features (X) and target (y)
X = df.drop(columns=['charges'])  # Assuming 'charges' is the target variable
y = df['charges']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model
lr_model = LinearRegression().fit(X_train_scaled, y_train) #Base
lr_lasso = Lasso().fit(X_train_scaled, y_train) #Lasso / L1 Regularization
lr_ridge = Ridge().fit(X_train_scaled, y_train) #Ridge / L2 Regularization

# Define the hyperparameters (alpha values) to test
param_grid = {'alpha': [0.01, 0.1, 1, 10]}

# Use GridSearchCV to find the best alpha
lasso_cv = GridSearchCV(lr_lasso, param_grid, cv=5)  # 5-fold cross-validation
lasso_cv.fit(X_train_scaled, y_train)
ridge_cv = GridSearchCV(lr_ridge, param_grid, cv=5)  # 5-fold cross-validation
ridge_cv.fit(X_train_scaled, y_train)

# Best alpha value
print(f"Best Alpha for Lasso: {lasso_cv.best_params_['alpha']}")
print(f"Best Alpha for Ridge: {ridge_cv.best_params_['alpha']}")

# Make predictions on the test set
y_pred = lr_model.predict(X_test_scaled)

# Make predictions with the best Lasso model
y_pred_lasso = lasso_cv.predict(X_test_scaled)

# Make predictions with the best Ridge model
y_pred_ridge = ridge_cv.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Print evaluation metrics
print(f"Test Score: {lr_model.score(X_test_scaled, y_test)}")
print(f"Training Score: {lr_model.score(X_train_scaled, y_train)}")
print(f"Linear Regression (After Feature Engineering) - Mean Squared Error: {mse}")
print(f"Linear Regression (After Feature Engineering) - R-squared: {r2}")
print(f"Lasso Regression (After Feature Engineering)- Mean Squared Error: {mse_lasso}")
print(f"Lasso Regression (After Feature Engineering)- R-squared: {r2_lasso}")
print(f"Ridge Regression (After Feature Engineering) - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression (After Feature Engineering) - R-squared: {r2_ridge}")

plt.figure(figsize=(24, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.grid()

# Lasso Regression
plt.subplot(1, 3, 2)
plt.scatter(y_test, y_pred_lasso, color='green', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Lasso Regression: Actual vs Predicted')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.grid()

# Ridge Regression
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred_ridge, color='purple', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Ridge Regression: Actual vs Predicted')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.grid()

plt.tight_layout()
plt.show()