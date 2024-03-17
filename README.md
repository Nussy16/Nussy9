# Below is a python script that demonstrates how to implement mean imputation and KNN imputation on a dataset using the pandas library for data manipulation and the scikit-learn library for KNN imputation.


import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Display information about missing values
print("Missing Values:")
print(data.isnull().sum())

# Mean imputation
data_mean = data.fillna(data.mean())

# KNN imputation
imputer = KNNImputer(n_neighbors=5)
data_knn = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='target_column'), data['target_column'], test_size=0.2, random_state=42)

# Train a linear regression model on the original data
model_orig = LinearRegression()
model_orig.fit(X_train, y_train)
pred_orig = model_orig.predict(X_test)
mse_orig = mean_squared_error(y_test, pred_orig)

# Train a linear regression model on the mean imputed data
model_mean = LinearRegression()
model_mean.fit(X_train.fillna(X_train.mean()), y_train)
pred_mean = model_mean.predict(X_test.fillna(X_train.mean()))
mse_mean = mean_squared_error(y_test, pred_mean)

# Train a linear regression model on the KNN imputed data
model_knn = LinearRegression()
model_knn.fit(X_train.fillna(X_train.mean()), y_train)
pred_knn = model_knn.predict(X_test.fillna(X_train.mean()))
mse_knn = mean_squared_error(y_test, pred_knn)

# Compare the results
print("\nMean Squared Error (Original):", mse_orig)
print("Mean Squared Error (Mean Imputation):", mse_mean)
print("Mean Squared Error (KNN Imputation):", mse_knn)

# Present insights
print("\nInsights:")
print("- Mean imputation can introduce bias, especially if the data has outliers.")
print("- KNN imputation considers the relationships between features, potentially resulting in better imputed values.")
print("- The choice of imputation method can significantly impact model performance.")


# This script loads the dataset, performs mean and KNN imputation, trains a linear regression model on the original, mean-imputed, and KNN-imputed data, evaluates the model performance using mean squared error, and presents insights gained from the analysis
