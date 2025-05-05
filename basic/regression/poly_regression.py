# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Load the Fish dataset
file_path = "Fish.csv"
df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS,
  "vipullrathod/fish-market",
  file_path,)
df.info()

# Print all features
print("Original features in the dataset:")
print(df.columns.tolist())
print("First few rows of the original dataset:")
print(df.head())

# Select features and target
features_to_keep = ['Length1', 'Length2', 'Length3', 'Height', 'Width']  # Exclude Species and Weight
X = df[features_to_keep]
y = df['Weight']

# Print selected features
print("\nSelected features for modeling:")
print(X.columns.tolist())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the polynomial regression model (degree=2)
degree = 2
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_train, y_train)

# Make predictions
y_pred = polyreg.predict(X_test)

# Calculate and print performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# # Extract the LinearRegression model from the pipeline
# lr_model = polyreg.named_steps['linearregression']
# poly_features = polyreg.named_steps['polynomialfeatures']

# # Get feature names after polynomial transformation
# feature_names = poly_features.get_feature_names_out(input_features=features_to_keep)

# # Print feature coefficients
# coefficients = pd.DataFrame({
#     'Feature': feature_names,
#     'Coefficient': lr_model.coef_
# })
# print("\nFeature Coefficients:")
# print(coefficients)