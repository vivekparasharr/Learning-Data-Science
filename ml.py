
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/01_pre_processing_data.csv')

# Separating independent and dependent variables
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Data Preprocessing
'''Generally, you want to treat the test set as though you did not have it during training. 
Whatever transformations you do to the train set should be done to the test set before you make predictions. 
If you apply transformation before splitting and then split into train/test you are leaking data from your test set (that is supposed to be completely withheld) into your training set. 
This will yield extremely biased results on model performance.'''

# Impute missing values - done after train-test-splitting to prevent data leakage
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:, 1:3])
X_train[:, 1:3] = imputer.transform(X_train[:, 1:3])

# Dummy coding the Independent Variable - we are using one hot encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train))

# Dummy coding the Dependent Variable - we are using label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Feature Scaling - 3 different techniques
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])

'''note - standard scalar expects a 2d array as input, so if the input has just 1 column, then we need to reshape it to a 2d array
y = y.reshape(len(y),1)'''

'''normalization/min-max-scaling - values in a column are shifted so that they are bound between 0 and 1
standardization - values in a column are rescaled to demonstrate properties of standard normal distribution (mean=0, variance=1)
standardization is preferred over normalization in most ML context, however neural network algorithms typically require data to be normalised to a 0 to 1 scale before model training

which classes of ml models require feature scaling?
Gradient descent based algorithms - scaling is required
- an iterative optimisation algorithm that takes us to the minimum of a function
- algorithms like linear regression and logistic regression rely on gradient descent to minimise their loss functions or in other words, to reduce the error between the predicted values and the actual values
Distance-based algorithms - scaling is required
- Algorithms like k-nearest neighbours, support vector machines and k-means clustering use the distance between data points to determine their similarity
Tree-based algorithms - scaling is NOT required
- Each node in a classification and regression trees (CART) model, otherwise known as decision trees represents a single feature in a dataset.
- The tree splits each node in such a way that it increases the homogeneity of that node. This split is not affected by the other features in the dataset.
'''

# REGRESSION 

# Simple Linear Regression

# Importing the dataset
dataset = pd.read_csv('data/02_Salary_Data.csv')

# Data preprocessing and transformaiton steps

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising actual (x axis) vs predicted (y axis) test set values
plt.scatter(y_test, y_pred) 


# Multiple Linear Regression

# Importing the dataset
dataset = pd.read_csv('data/02_50_Startups.csv')

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


# Polynomial Regression

# Importing the dataset
dataset = pd.read_csv('data/02_Position_Salaries.csv')

# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # we are chosing n=4
X_poly = poly_reg.fit_transform(X) # transform single feature matrix X to a matrix of features X^1, X^2, .. X^n
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Predicting the Test set results
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))



# Support Vector Regression (SVR)

# Importing the dataset
dataset = pd.read_csv('data/02_Position_Salaries.csv')

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


# Decision Tree Regression

dataset = pd.read_csv('data/02_Position_Salaries.csv')

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])



# Random Forest Regression

dataset = pd.read_csv('data/02_Position_Salaries.csv')

# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

