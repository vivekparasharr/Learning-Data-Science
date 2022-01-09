
#########################################################
##################### Preprocessing #####################
#########################################################

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


##############################################################
#################  ARTIFICIAL NEURAL NETWORK  ################
##############################################################


# Artificial Neural Network for regression

import numpy as np
import pandas as pd
import tensorflow as tf

tf.__version__

# Part 1 - Data Preprocessing
dataset = pd.read_excel('data/10_Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Building the ANN
# Initializing the ANN
ann = tf.keras.models.Sequential()
# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1))

# Training the ANN
# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Training the ANN model on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Predicting the results of the Test set
y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))






# Artificial Neural Network for classificaiton

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__

# Part 1 - Data Preprocessing

# Importing the dataset
dataset = pd.read_csv('data/10_Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y)

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN

# Initializing the ANN
# Allows to build the neural network as a sequence of layers (neurons connected in successive layers) as opposed to a computational graph (neurons connected any way, example - Boltzmann machines, restricted or deep Boltzmann machines)
# Since tensorflow 2.0, keras library is integrated in tensorflow. Before this they used to be separate libraries
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Part 3 - Training the ANN

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)

# Part 4 - Making the predictions and evaluating the model

# Predicting the result of a single observation

"""
Homework:
Use our ANN model to predict if the customer with the following informations will leave the bank: 
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $ 60000
Number of Products: 2
Does this customer have a credit card? Yes
Is this customer an Active Member: Yes
Estimated Salary: $ 50000
So, should we say goodbye to that customer?

Solution:
"""

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)

"""
Therefore, our ANN model predicts that this customer stays in the bank!
Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.
"""

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)




