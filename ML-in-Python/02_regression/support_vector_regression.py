# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values #2-d array
y = dataset.iloc[:, -1].values #1-d vector

# first we need to convert this 1-d vector to a 2-d array
# because standard scale class expects a 2-d array as input, and we are going to use this for feature scaling
y=y.reshape(len(y),1) #reshape(number of rows, number of cols), we wants the number of rows to be the length of the vector 

# apply feature scaling
# we need to apply feature scaling because there are no coefficients (like in linear reg) that can compensate for the high value of the features
# for models that have an imlicit equation/relationship between dependent and independent variables, we need to do feature scaling
# we need two separate sc objects because sc is based on the mean and sd of the variable that is fed to it and since X and y variables have different means and sd's we need to separate sc's
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X=sc_X.fit_transform(X)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # We can chose the kernel. Some kernel can learn linear and some non linear relationships
regressor.fit(X, y)

# Predicting a new result
# lets say we want to predict y when X=6.5
X=6.5
X=[[6.5]] #first, we convert X into a 2-d array because regressor.predict function expect a 2-d array as input
X=sc_X.transform(X) #second, we scale X that we want to predict because our model is built on scaled values of X which we did using sc_X scaler
y=regressor.predict(X) #third, we predict the value of y; keep in mind that this predicted value of y is in the scale that was applied to y
y=sc_y.inverse_transform(y) #fourth, we reverse the scaling that we applied on y using sc_y

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') # we show X and actual y; this will give us the real points in their original scale, we need to do this because we applied transforms to X and y earlier
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color = 'blue') # we show X and predicted y; we use the regressor function to predict y values and then the sc_y.inverse_transform to unscale them
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
