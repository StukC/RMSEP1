# Import necessary libraries
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Function to load data
def load_data(file_name):
    return np.loadtxt(file_name, delimiter=' ')

# Load training and test data
train_data = load_data('train.dat')
test_data = load_data('test.dat')

# Separate features and targets
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Normalize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))
X_test_scaled = scaler.transform(X_test.reshape(-1, 1))

# Initialize variables for tracking the best model
best_degree = 0
best_rmse = float('inf')
best_model = None

# Range of degrees to try for our polynomial
for degree in range(1, 25):
    # Generate polynomial features
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    # Initialize Ridge Regression model
    model = Ridge(alpha=1.0)

    # Perform cross-validation
    scores = cross_val_score(model, X_train_poly, y_train, scoring='neg_root_mean_squared_error', cv=5)
    avg_rmse = -scores.mean()

    # Update the best model if current model is better
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_degree = degree
        best_model = model

# Train the best model on the entire training set
poly = PolynomialFeatures(degree=best_degree)
X_train_poly = poly.fit_transform(X_train_scaled)
best_model.fit(X_train_poly, y_train)

# Predict and calculate RMSE on the test set
X_test_poly = poly.transform(X_test_scaled)
y_pred = best_model.predict(X_test_poly)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print the best degree and RMSE
print(f"Best Polynomial Degree: {best_degree}")
print(f"Test RMSE: {rmse}")

# Plotting the results
plt.scatter(X_train, y_train, color='red', label='Training Data')
X_plot = np.linspace(X_train.min(), X_train.max(), 100)
X_plot_scaled = scaler.transform(X_plot.reshape(-1, 1))
X_plot_poly = poly.transform(X_plot_scaled)
y_plot = best_model.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color='blue', label='Polynomial Curve')
plt.xlabel('Week')
plt.ylabel('New COVID-19 Cases')
plt.title('Polynomial Regression on COVID-19 Cases Data')
plt.legend()
plt.show()
