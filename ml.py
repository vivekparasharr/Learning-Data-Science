
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



# Hierarchical clustering

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




#########################################################
############## ASSOCIATION RULE LEARNING ################
#########################################################

# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori
# each row represents a transaction. in the columns we have the products that were purchased
dataset = pd.read_csv('data/05_Market_Basket_Optimisation.csv', header = None)

# converting dataset to a list (transactions) of lists (products purchased in transactions)
# apriori algo expects data item strings in the list of list 
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame

## Displaying the results sorted by descending lifts
resultsinDataFrame.nlargest(n = 10, columns = 'Lift')



# Eclat

# Run the following command in the terminal to install the apyori package: pip install apyori
# Data Preprocessing
dataset = pd.read_csv('data/05_Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

# Training the Eclat model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])

## Displaying the results sorted by descending supports
resultsinDataFrame.nlargest(n = 10, columns = 'Support')






#########################################################
############## reinforcement LEARNING ################
#########################################################


# Upper Confidence Bound (UCB)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data/06_Ads_CTR_Optimisation.csv')

# Implementing UCB
import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()



