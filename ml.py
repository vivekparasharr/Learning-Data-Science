
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

#########################################################
###################### REGRESSION #######################
#########################################################

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



#########################################################
#################### CLASSIFICATION #####################
#########################################################


# Logistic Regression

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
classifier.predict(sc.transform([[30,87000]]))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)


# K-Nearest Neighbors (K-NN)

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)

# Accuracy
accuracy_score(y_test, y_pred)



# Support Vector Machine (SVM)

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)




# Kernel SVM

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)



# Naive Bayes

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)




# Decision Tree Classification

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)





# Random Forest Classification

# Importing the dataset
dataset = pd.read_csv('data/03_Social_Network_Ads.csv')

# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
print(classifier.predict(sc.transform([[30,87000]])))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)




#########################################################
####################### CLUSTERING ######################
#########################################################

# K-Means Clustering

# Importing the dataset
dataset = pd.read_csv('data/04_Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y_kmeans.shape

# Visualising the clusters
for i, c_ in zip(range(0,5),['r','b','g','c','m']):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s = 100, c = c_, label = 'Cluster '+str(i))
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()




# Importing the dataset
dataset = pd.read_csv('data/04_Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Training the model
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualizing the clusters
for i, c_ in zip(range(0,5),['r','b','g','c','m']):
    plt.scatter(X[y_hc == i, 0], X[y_hc == i, 1], s = 100, c = c_, label = 'Cluster '+str(i))
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


