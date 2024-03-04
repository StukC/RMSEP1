# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Load the training and test data
train_data = np.loadtxt("train.dat")
test_data = np.loadtxt("test.dat")

# Separate the input features and output values for training and test data
X_train_raw = train_data[:, 0].reshape(-1, 1)
y_train_raw = train_data[:, 1]
X_test_raw = test_data[:, 0].reshape(-1, 1)
y_test_raw = test_data[:, 1]

# Normalize the features and target using StandardScaler
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train_raw)
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
X_test_scaled = scaler_X.transform(X_test_raw)
y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()

# Define the cross-validation strategy
kf = KFold(n_splits=12, shuffle=False)

# Initialize lists to store cross-validation RMSE results
cv_train_rmse = []
cv_test_rmse = []

# Loop over degrees of polynomial from 0 to 28
for degree in range(1, 29):
    # Create a pipeline with PolynomialFeatures and Ridge regression
    pipeline = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=0, fit_intercept=True))
    
    # Initialize lists to store RMSE for each fold
    fold_train_rmse = []
    fold_test_rmse = []
    
    # Perform cross-validation
    for train_index, test_index in kf.split(X_train_scaled):
        # Split the data into training and test sets for this CV fold
        X_cv_train, X_cv_test = X_train_scaled[train_index], X_train_scaled[test_index]
        y_cv_train, y_cv_test = y_train_scaled[train_index], y_train_scaled[test_index]
        
        # Fit the model to the training data of this CV fold
        pipeline.fit(X_cv_train, y_cv_train)
        
        # Predict on the training and test data of this CV fold
        y_cv_train_pred = pipeline.predict(X_cv_train)
        y_cv_test_pred = pipeline.predict(X_cv_test)
        
        # Calculate RMSE for this fold and append to lists
        fold_train_rmse.append(np.sqrt(mean_squared_error(y_cv_train, y_cv_train_pred)))
        fold_test_rmse.append(np.sqrt(mean_squared_error(y_cv_test, y_cv_test_pred)))
    
    # Append average RMSE across folds to the cv_train_rmse and cv_test_rmse lists
    cv_train_rmse.append(np.mean(fold_train_rmse))
    cv_test_rmse.append(np.mean(fold_test_rmse))

# Find the degree with the lowest average test RMSE
best_degree = np.argmin(cv_test_rmse)
print(f"Best polynomial degree: {best_degree}")

# Train the final model on the entire training set with the best degree
final_pipeline = make_pipeline(PolynomialFeatures(degree=best_degree, include_bias=False), Ridge(alpha=0, fit_intercept=True))
final_pipeline.fit(X_train_scaled, y_train_scaled)

# Save the coefficients for the best model
final_coeffs = final_pipeline.named_steps['ridge'].coef_
np.savetxt("w.dat", final_coeffs)

# Predict on the test data using the final model
y_test_pred_scaled = final_pipeline.predict(X_test_scaled)

# Transform the predictions back to the original scale
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# Calculate the RMSE on the test data
test_rmse = np.sqrt(mean_squared_error(y_test_raw, y_test_pred))
print(f"Test RMSE: {test_rmse}")

# Calculate the RMSE on the training data for the final model
y_train_pred_scaled = final_pipeline.predict(X_train_scaled)
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
train_rmse = np.sqrt(mean_squared_error(y_train_raw, y_train_pred))
print(f"Train RMSE: {train_rmse}")

# Generate a range of values from the minimum to the maximum training week numbers
week_range_scaled = np.linspace(X_train_scaled.min(), X_train_scaled.max(), 500).reshape(-1, 1)

# Predict the number of cases for our range of week numbers using the final model
predicted_cases_scaled = final_pipeline.predict(week_range_scaled)

# Inverse transform the predicted scaled number of cases to the original number of cases
predicted_cases = scaler_y.inverse_transform(predicted_cases_scaled.reshape(-1, 1)).flatten()

# Inverse transform the scaled week range to the original week numbers
week_range = scaler_X.inverse_transform(week_range_scaled).flatten()

# Plot the training data
plt.scatter(X_train_raw, y_train_raw, color='blue', label='Training data')

# Plot the test data
plt.scatter(X_test_raw, y_test_raw, color='yellow', label='Test data')

# Plot the polynomial curve
plt.plot(week_range, predicted_cases, color='red', label=f'Polynomial Degree {best_degree}')

# Label the axes
plt.xlabel('Week Number')
plt.ylabel('Number of New COVID-19 Cases')

# Add a title
plt.title('Polynomial Regression for COVID-19 Cases')

# Add a legend
plt.legend()

# Show the plot
plt.show()

# Plot the average training RMSE vs. polynomial degree
degrees = list(range(1, 29))  # Assuming you started your degrees at 1
plt.plot(degrees, cv_train_rmse, color='blue', marker='o', label='Average CV Train RMSE')  # 'marker' specifies the style of the points
plt.scatter(degrees, cv_train_rmse, color='red')  # This adds the actual points on the plot
plt.xlabel('Polynomial Degree')
plt.ylabel('Average CV Train RMSE')
plt.title('Average CV Train RMSE vs Polynomial Degree')
plt.legend()
plt.show()
