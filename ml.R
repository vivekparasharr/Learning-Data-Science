
# Data Preprocessing Template

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



