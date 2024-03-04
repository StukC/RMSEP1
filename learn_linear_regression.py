# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Load the training and test data
trainData = np.loadtxt("train.dat")
testData = np.loadtxt("test.dat")

# Separate the input and output values for training and test data
xTrainRaw = trainData[:, 0].reshape(-1, 1)
yTrainRaw = trainData[:, 1]
xTestRaw = testData[:, 0].reshape(-1, 1)
yTestRaw = testData[:, 1]

# Normalize the features and target using StandardScaler
xScaler = StandardScaler()
yScaler = StandardScaler()

xTrainScaled = xScaler.fit_transform(xTrainRaw)
yTrainScaled = yScaler.fit_transform(yTrainRaw.reshape(-1, 1)).flatten()
xTestScaled = xScaler.transform(xTestRaw)
yTestScaled = yScaler.transform(yTestRaw.reshape(-1, 1)).flatten()

kf = KFold(n_splits=12, shuffle=False)

# Initialize lists to store cross-validation RMSE results
cvTrainRMSE = []
cvTestRMSE = []

# polynomial degree loop
for degree in range(1, 29):

    pipeline = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), Ridge(alpha=0, fit_intercept=True))
    foldTrainRMSE = []
    foldTestRMSE = []
    
    # cross validation
    for trainIndex, testIndex in kf.split(xTrainScaled):
        # Split the data
        xcvTrain, xcvTest = xTrainScaled[trainIndex], xTrainScaled[testIndex]
        ycvTrain, ycvTest = yTrainScaled[trainIndex], yTrainScaled[testIndex]
        
        # Fit the model to the training data of CV fold
        pipeline.fit(xcvTrain, ycvTrain)
        
        # Predict on the training and test data of CV fold
        ycvTrainPrediction = pipeline.predict(xcvTrain)
        ycvTestPrediction = pipeline.predict(xcvTest)
        
        # Calculate RMSE and append
        foldTrainRMSE.append(np.sqrt(mean_squared_error(ycvTrain, ycvTrainPrediction)))
        foldTestRMSE.append(np.sqrt(mean_squared_error(ycvTest, ycvTestPrediction)))
    
    # Append average RMSE across folds
    cvTrainRMSE.append(np.mean(foldTrainRMSE))
    cvTestRMSE.append(np.mean(foldTestRMSE))

# Find the degree with the lowest average test RMSE
bestDegree = np.argmin(cvTestRMSE)
print(f"Best polynomial degree: {bestDegree}")

# Train the final model on the entire training set with the best degree
finalPipeline = make_pipeline(PolynomialFeatures(degree=bestDegree, include_bias=False), Ridge(alpha=0, fit_intercept=True))
finalPipeline.fit(xTrainScaled, yTrainScaled)

# Save the coefficients
finalCoefficients = finalPipeline.named_steps['ridge'].coef_
np.savetxt("w.dat", finalCoefficients)

# Predict on test data using final model
yTestPredictionScaled = finalPipeline.predict(xTestScaled)

# Transform the predictions back to OG scale
yTestPrediction = yScaler.inverse_transform(yTestPredictionScaled.reshape(-1, 1)).flatten()

# Calculate the RMSE on test data
testRMSE = np.sqrt(mean_squared_error(yTestRaw, yTestPrediction))
print(f"Test RMSE: {testRMSE}")

# Calculate RMSE on the training data for final model
yTrainPredictionScaled = finalPipeline.predict(xTrainScaled)
yTrainPrediction = yScaler.inverse_transform(yTrainPredictionScaled.reshape(-1, 1)).flatten()
trainRMSE = np.sqrt(mean_squared_error(yTrainRaw, yTrainPrediction))
print(f"Train RMSE: {trainRMSE}")

# Predictions
weekRangeScaled = np.linspace(xTrainScaled.min(), xTrainScaled.max(), 91).reshape(-1, 1)
predictedCasesScaled = finalPipeline.predict(weekRangeScaled)

# Inverse transform 
predictedCases = yScaler.inverse_transform(predictedCasesScaled.reshape(-1, 1)).flatten()

# Inverse transform
weekRange = xScaler.inverse_transform(weekRangeScaled).flatten()

# Plot training data
plt.scatter(xTrainRaw, yTrainRaw, color='blue', label='Training data')

# Plot test data
plt.scatter(xTestRaw, yTestRaw, color='yellow', label='Test data')

# Plot polynomial curve
plt.plot(weekRange, predictedCases, color='red', label=f'Polynomial Degree {bestDegree}')

# Label axis
plt.xlabel('Week Number')
plt.ylabel('Number of New COVID-19 Cases')

# Title
plt.title('Polynomial Regression for COVID-19 Cases')

# Legend
plt.legend()

# Show plot
plt.show()

# Plot the average training RMSE vs. polynomial degree
degrees = list(range(1, 29))
plt.plot(degrees, cvTrainRMSE, color='blue', marker='o', label='Average CV Train RMSE')
plt.scatter(degrees, cvTrainRMSE, color='red')
plt.xlabel('Polynomial Degree')
plt.ylabel('Average CV Train RMSE')
plt.title('Average CV Train RMSE vs Polynomial Degree')
plt.legend()
plt.show()
