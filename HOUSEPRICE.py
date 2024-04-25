import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('data.csv')

# Remove rows with price = 0
df = df[df['price'] != 0]

# Prepare the data
X = df['sqft_living'].values.reshape(-1, 1)
y = df['price'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)

# Ridge Regression
alpha = 1.0  # Regularization strength
model_ridge = Ridge(alpha=alpha)
model_ridge.fit(X_train, y_train)
y_pred_ridge = model_ridge.predict(X_test)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)

# Gradient Boosting Regression
model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
mae_gb = mean_absolute_error(y_test, y_pred_gb)
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)

# Plotting
info_dict = {'Title': 'House Prices', 'X_label': 'Property Size (sq ft)', 'y_label': 'Price (AUD)'}

plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred_lr, color='red', label='Linear Regression')
plt.plot(X_test, y_pred_ridge, color='green', label='Ridge Regression')
plt.plot(X_test, y_pred_gb, color='purple', label='Gradient Boosting Regression')
plt.title(info_dict['Title'])
plt.xlabel(info_dict['X_label'])
plt.ylabel(info_dict['y_label'])
plt.legend()
plt.show()

# Print evaluation metrics
print("Linear Regression:")
print("  Mean Absolute Error:", mae_lr)
print("  Root Mean Squared Error:", rmse_lr)
print("Ridge Regression:")
print("  Mean Absolute Error:", mae_ridge)
print("  Root Mean Squared Error:", rmse_ridge)
print("Gradient Boosting Regression:")
print("  Mean Absolute Error:", mae_gb)
print("  Root Mean Squared Error:", rmse_gb)


# Apply polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train the final model
final_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
final_model.fit(X_poly, y)

# User input for relevant features
user_input = {
    'sqft_living': float(input("Enter property size in sq ft: "))
}
bedrooms: int(input("Enter number of bedrooms: "))
bathrooms: float(input("Enter number of bathrooms: "))
floors: float(input("Enter number of floors: "))
zipcode: float(input("Enter pincode (zipcode): "))
# Transform user input using polynomial features
user_input_poly = poly.transform(np.array([[user_input['sqft_living']]]))

# Predict using the final model
predicted_price = final_model.predict(user_input_poly)[0]

print(f"Predicted price for a property of size {user_input['sqft_living']:.2f} sq ft: ${predicted_price:.2f}")