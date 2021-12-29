
#########################################################
##################### Preprocessing #####################
#########################################################

# Importing the dataset
dataset = read.csv('02_salary_data.csv')

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Encoding categorical data
dataset$State = factor(dataset$State, levels = c('New York', 'California', 'Florida'), labels = c(1, 2, 3))

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)




#########################################################
###################### REGRESSION #######################
#########################################################


# Simple Linear Regression
# Fitting Simple Linear Regression to the Training set
regressor = lm(formula = Salary ~ YearsExperience, data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

# Predicting a new result with Linear Regression
predict(regressor, data.frame(Level = 6.5))

# Visualising the Training set results
# install.packages('ggplot2')
library(ggplot2)
ggplot() + geom_point(aes(x = test_set$Salary, y = y_pred))



# Multiple Linear Regression

# Importing the dataset
dataset = read.csv('02_50_Startups.csv')

# Fitting Multiple Linear Regression to the Training set
regressor = lm(formula = Profit ~ ., data = training_set)

# Predicting the Test set results
y_pred = predict(regressor, newdata = test_set)

ggplot() + geom_point(aes(x = test_set$Profit, y = y_pred))





# Polynomial Regression

# Importing the dataset
dataset = read.csv('02_Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ ., data = dataset)

# Fitting Polynomial Regression to the dataset
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
poly_reg = lm(formula = Salary ~ ., data = dataset)

# Predicting
y_pred = predict(poly_reg, newdata = dataset)

# Predicting a new result with Polynomial Regression
predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))

# Visualizing
ggplot() + geom_point(aes(x = dataset$Salary, y = y_pred))






# SVR

# Importing the dataset
dataset = read.csv('02_Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting SVR to the dataset
# install.packages('e1071')
library(e1071)
regressor = svm(formula = Salary ~ ., data = dataset, type = 'eps-regression', kernel = 'radial')

# Predicitng values of dataset
y_pred = predict(regressor, newdata = dataset)

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualizing
ggplot() + geom_point(aes(x = dataset$Salary, y = y_pred))







# Decision Tree Regression

# Importing the dataset
dataset = read.csv('02_Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting the model
library(rpart)
regressor = rpart(formula = Salary ~ ., data = dataset, control = rpart.control(minsplit = 1))

# Predicting
y_pred = predict(regressor, newdata = dataset)

# Predicting a new result with Decision Tree Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising 
ggplot() + geom_point(aes(x = dataset$Salary, y = y_pred))

# Plotting the tree
plot(regressor)
text(regressor)





# Random Forest Regression

# Importing the dataset
dataset = read.csv('02_Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting Random Forest Regression to the dataset
# install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[-2], y = dataset$Salary, ntree = 500)

# Predicting  
y_pred = predict(regressor, newdata = dataset)

# Predicting a new result with Random Forest Regression
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising 
ggplot() + geom_point(aes(x = dataset$Salary, y = y_pred))





#########################################################
#################### CLASSIFICATION #####################
#########################################################


# Logistic Regression

# Importing the dataset
dataset = read.csv('03_Social_Network_Ads.csv')

# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting Logistic Regression to the Training set
classifier = glm(formula = Purchased ~ ., family = binomial, data = training_set)

# Predicting the Test set results
prob_pred = predict(classifier, type = 'response', newdata = test_set[-3])
y_pred = ifelse(prob_pred > 0.5, 1, 0)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred > 0.5)




# K-Nearest Neighbors (K-NN)

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting K-NN to the Training set and Predicting the Test set results
library(class)
y_pred = knn(train = training_set[, -3], test = test_set[, -3], cl = training_set[, 3], k = 5, prob = TRUE)

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)



# Support Vector Machine (SVM)

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting SVM to the Training set
# install.packages('e1071')
library(e1071)
classifier = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)





# Kernel SVM

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting Kernel SVM to the Training set
library(e1071)
classifier = svm(formula = Purchased ~ ., data = training_set, type = 'C-classification', kernel = 'radial')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)




# Naive Bayes

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting naive bayes to the Training set
# install.packages('e1071')
library(e1071)
classifier = naiveBayes(x = training_set[-3], y = training_set$Purchased)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)




# Decision Tree Classification

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting Decision Tree Classification to the Training set
library(rpart)
classifier = rpart(formula = Purchased ~ ., data = training_set)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3], type = 'class')

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)



# Random Forest Classification

# Importing the dataset
# Encoding the target feature as factor
# Splitting the dataset into the Training set and Test set
# Feature Scaling

# Fitting Random Forest Classification to the Training set
# install.packages('randomForest')
library(randomForest)
set.seed(123)
classifier = randomForest(x = training_set[-3], y = training_set$Purchased, ntree = 500)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)



#########################################################
####################### CLUSTERING ######################
#########################################################


# K-Means Clustering

# Importing the dataset
dataset = read.csv('04_Mall_Customers.csv')
dataset = dataset[4:5]

# Using the elbow method to find the optimal number of clusters
set.seed(6)
wcss = vector()
for (i in 1:10) wcss[i] = sum(kmeans(dataset, i)$withinss)
plot(1:10, wcss, type = 'b', main = paste('The Elbow Method'), xlab = 'Number of clusters', ylab = 'WCSS')

# Fitting K-Means to the dataset
set.seed(29)
kmeans = kmeans(x = dataset, centers = 5)
y_kmeans = kmeans$cluster

# Visualising the clusters
library(cluster)
clusplot(dataset, y_kmeans, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE, span = TRUE, main = paste('Clusters of customers'), xlab = 'Annual Income', ylab = 'Spending Score')



# Hierarchical Clustering

# Using the dendrogram to find the optimal number of clusters
dendrogram = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
plot(dendrogram, main = paste('Dendrogram'), xlab = 'Customers', ylab = 'Euclidean distances')

# Fitting Hierarchical Clustering to the dataset
hc = hclust(d = dist(dataset, method = 'euclidean'), method = 'ward.D')
y_hc = cutree(hc, 5)

# Visualising the clusters
library(cluster)
clusplot(dataset, y_hc, lines = 0, shade = TRUE, color = TRUE, labels= 2, plotchar = FALSE, span = TRUE, main = paste('Clusters of customers'), xlab = 'Annual Income', ylab = 'Spending Score')





#########################################################
############## ASSOCIATION RULE LEARNING ################
#########################################################






