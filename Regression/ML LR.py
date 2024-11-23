import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso,Ridge
from sklearn.model_selection import GridSearchCV

#EDA for Data
'''
df = pd.read_csv("C:/Users/User/Downloads/insurance.csv")

# List of numerical columns to check for outliers 
numerical_columns = ['age', 'bmi', 'children', 'charges'] 

# Create boxplots for each numerical feature
plt.figure(figsize=(15, 10))

for i, col in enumerate(numerical_columns, 1):
    plt.subplot(2, 2, i)  # Change (2, 2) depending on how many subplots you want in a grid
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

#Histogram
plt.figure(figsize=(15, 10))
sns.histplot(df['charges'], kde=True)
plt.title('Distribution of Charges')
plt.show()

#Binary encoding for 'sex' and 'smoker'
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

#One-hot encoding for 'region'
df = pd.get_dummies(df, columns=['region'], drop_first=False)

#correlation_matrix = df.corr()
heatmap = px.imshow(correlation_matrix,labels=dict(color="Correlation"), title='Correlation Heatmap', text_auto=True)
heatmap.show()
'''

#Data Preprocessing For Linear Regression
'''
df = pd.read_csv("C:/Users/User/Downloads/insurance.csv")
df = df.drop(columns=['region'])

# Binary encoding for 'sex' and 'smoker'
df['sex'] = df['sex'].map({'male': 1, 'female': 0})
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
df.to_csv('C:/Users/User/Downloads/insurance_linear.csv', index=False)
'''

df = pd.read_csv("C:/Users/User/Downloads/insurance_linear.csv")

# Define features (X) and target (y)
X = df.drop(columns=['charges'])
y = df['charges']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialise the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the linear regression model on scaled data
lr_model = LinearRegression().fit(X_train_scaled, y_train)
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
print(f"Linear Regression - Mean Squared Error: {mse}")
print(f"Linear Regression - R-squared: {r2}")
print(f"Lasso Regression - Mean Squared Error: {mse_lasso}")
print(f"Lasso Regression - R-squared: {r2_lasso}")
print(f"Ridge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - R-squared: {r2_ridge}")

# Plot Actual vs Predicted for Linear, Lasso, and Ridge side by side
plt.figure(figsize=(18, 6))

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
