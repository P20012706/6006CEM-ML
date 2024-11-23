import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

# Data Preprocessing
df = pd.read_csv("C:/Users/User/Downloads/loan_approval_processed.csv")
'''
# Drop the loan_id column as it's not needed
df.drop(columns=['loan_id'], inplace=True)

# One-hot encode categorical columns
df['education'] = df['education'].map({' Graduate': 1, ' Not Graduate': 0})
df['self_employed'] = df['self_employed'].map({' Yes': 1, ' No': 0})
df['loan_status'] = df['loan_status'].map({' Approved': 1, ' Rejected': 0})

df.to_csv('C:/Users/User/Downloads/loan_approval_processed.csv', index=False)


# Display basic information about the dataset
print("Dataset Head:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

#Data Visualization (EDA)

# Correlation heatmap (excluding loan_id)
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Correlation Heatmap")
plt.show()



#Boxplots for numerical features to check for outliers
plt.figure(figsize=(12, 10))

# Boxplot for no_of_dependents
plt.subplot(3, 3, 1)
sns.boxplot(data=df, x='no_of_dependents')
plt.title('Boxplot for no_of_dependents')

# Boxplot for income_annum
plt.subplot(3, 3, 2)
sns.boxplot(data=df, x='income_annum')
plt.title('Boxplot for income_annum')

# Boxplot for loan_amount
plt.subplot(3, 3, 3)
sns.boxplot(data=df, x='loan_amount')
plt.title('Boxplot for loan_amount')

# Boxplot for loan_term
plt.subplot(3, 3, 4)
sns.boxplot(data=df, x='loan_term')
plt.title('Boxplot for loan_term')

# Boxplot for cibil_score
plt.subplot(3, 3, 5)
sns.boxplot(data=df, x='cibil_score')
plt.title('Boxplot for cibil_score')

# Boxplot for residential_assets_value
plt.subplot(3, 3, 6)
sns.boxplot(data=df, x='residential_assets_value')
plt.title('Boxplot for residential_assets_value')

# Boxplot for luxury_assets_value
plt.subplot(3, 3, 7)
sns.boxplot(data=df, x='luxury_assets_value')
plt.title('Boxplot for luxury_assets_value')

# Boxplot for bank_asset_value
plt.subplot(3, 3, 8)
sns.boxplot(data=df, x='bank_asset_value')
plt.title('Boxplot for bank_asset_value')

plt.tight_layout()
plt.show()



# List of numerical columns to plot histograms for
numerical_columns = [
    'no_of_dependents', 'income_annum', 'loan_amount', 'loan_term', 
    'cibil_score', 'residential_assets_value', 'luxury_assets_value', 'bank_asset_value'
]

one_hot_columns = ['education', 'self_employed', 'loan_status']

plt.figure(figsize=(15, 12)) 
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  
    sns.histplot(df[col], kde=True)  
    plt.title(f"Histogram of {col}")
    plt.xlabel('') 
    plt.ylabel('Frequency')

    
plt.figure(figsize=(15, 5)) 
for i, col in enumerate(one_hot_columns, 1):  # Start with index 1 for proper subplot positioning
    plt.subplot(1, 3, i)  # 1 row, 3 columns for the plots
    sns.countplot(data=df, x=col)  # Plot bar plot for categorical features
    plt.title(f"Bar Plot of {col}")
    plt.xlabel('')  # Turn off x-axis label
    plt.ylabel('')  # Turn off y-axis label

plt.tight_layout()  # Ensure there's no overlap between plots
plt.show()
'''

X = df.drop(columns=["loan_status"]) 
y = df["loan_status"] 

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'criterion': ['gini', 'entropy']
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get best model and predictions
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

# Print best parameters and score
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Score: {grid_search.best_score_}")

feature_importances = best_rf_model.feature_importances_

# Create a DataFrame for feature names and their respective importance scores
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Evaluate the model
print("Feature Importance for Random Forest:")
print(feature_importance_df)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=["Rejected", "Approved"], 
            yticklabels=["Rejected", "Approved"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()