
import numpy as np
import pandas as pd
import scipy.stats as spss
import plotly.express as px
import seaborn as sns

'''
Central Limit Theorem

The central limit theorem states that the sampling distribution of the mean of any independent, random variable will be normal or nearly normal, if the sample size is large enough.

How large is "large enough"? The answer depends on two factors.

    Requirements for accuracy. The more closely the sampling distribution needs to resemble a normal distribution, the more sample points will be required.
    The shape of the underlying population. The more closely the original population resembles a normal distribution, the fewer sample points will be required.

In practice, some statisticians say that a sample size of 30 is large enough when the population distribution is roughly bell-shaped. Others recommend a sample size of at least 40. But if the original population is distinctly not normal (e.g., is badly skewed, has multiple peaks, and/or has outliers), researchers like the sample size to be even larger.
'''

'''
Continuous Distributions:
- Uniform distribution
- Normal Distribution, also known as Gaussian distribution
- Standard Normal Distribution - case of normal distribution where loc or mean = 0 and scale or sd = 1 
- Gamma distribution - exponential, chi-squared, erlang distributions are special cases of the gamma distribution
- Erlang distribution - special form of Gamma distribution when a is an integer ?
- Exponential distribution - special form of Gamma distribution with a=1
- Lognormal - not covered
- Chi-Squared - not covered
- Weibull - not covered
- t Distribution - not covered
- F Distribution - not covered

Discrete Distributions:
- Poisson distribution is a limiting case of a binomial distribution under the following conditions: n tends to infinity, p tends to zero and np is finite
- Binomial Distribution
- Negative Binomial - not covered
- Bernoulli Distribution is a special case of the binomial distribution where a single trial is conducted n=1
- Geometric - not covered
'''


# uniform distribution
# generate an array of random variables using scipy
# size specifies number of random variates, loc corresponds to mean, scale corresponds to standard deviation
rv_array = spss.uniform.rvs(size=10000, loc = 10, scale=20) 

# we can directly plot the data from the array
px.histogram(rv_array) # plotted using plotly express
sns.histplot(rv_array, kde=True) # plotted using seaborn


# or we can convert array into a dataframe and then plot the data frame
rv_df = pd.DataFrame(rv_array, columns=['value_of_random_variable'])
px.histogram(rv_df, x='value_of_random_variable', nbins=20) # plotted using plotly express
sns.histplot(data=rv_df, x='value_of_random_variable', kde=True) # plotted using seaborn


# Normal Distribution, also known as Gaussian distribution
# normal distribution has a bell-shaped density curve described by its mean and standard deviation. 
# The density curve is symmetrical, centered about its mean, with its spread determined by its standard deviation 
# showing that data near the mean are more frequent in occurrence than data far from the mean. 
'''
Normal distribution is a limiting case of Poisson distribution with the parameter lambda tends to infinity

Since normal distribution is a limiting case of Poisson distribution, it is also
another limiting form of binomial distribution under the following conditions:
- The number of trials is indefinitely large, n tends to infinity
- Both p and q are not indefinitely small.
'''
rv_array = spss.norm.rvs(size=10000,loc=10,scale=100)  # size specifies number of random variates, loc corresponds to mean, scale corresponds to standard deviation
sns.histplot(rv_array, kde=True)

# we can add x and y labels, change the number of bins, color of bars, etc.
# with distplot we can supply additional arguments for adjusting width of bars, transparency, etc.
ax = sns.distplot(rv_array, bins=100, kde=True, color='cornflowerblue', hist_kws={"linewidth": 15,'alpha':1})
ax.set(xlabel='Normal Distribution', ylabel='Frequency')


# Standard Normal Distribution
# Is a special case of the normal distribution where mean = 0 and sd = 1 
# z-score (aka, a standard score) indicates how many standard deviations an element is from the mean
# z = ( value of element - population mean ) / population sd
# if z score = 0 then the element = mean, if < 0 then element < mean and if > 0 then element is > mean
rv_array = spss.norm.rvs(size=10000,loc=0,scale=1)  # case of normal distribution where loc or mean = 0 and scale or sd = 1 
sns.histplot(rv_array, kde=True)


# Gamma distribution
# gamma distribution is a two-parameter family of continuous probability distributions
# it is used rarely in its raw form
# exponential, chi-squared, erlang distributions are special cases of the gamma distribution
rv_array = spss.gamma.rvs(a=5, size=10000) # size specifies number of random variates, a is the shape parameter
sns.distplot(rv_array, kde=True)

# Erlang distribution - special form of Gamma distribution when a is an integer
# Exponential distribution - special form of Gamma distribution with a=1

# Exponential distribution
# Exponential distribution - special form of Gamma distribution with a=1
# exponential distribution describes the time between events in a Poisson point process, 
# i.e., a process in which events occur continuously and independently at a constant average rate
rv_array = spss.expon.rvs(scale=1,loc=0,size=1000) # size specifies number of random variates, loc corresponds to mean, scale corresponds to standard deviation
sns.distplot(rv_array, kde=True)


# Poisson Distribution
'''
Poisson random variable is typically used to model the number of times an event happened in a time interval. 
For example, the number of users visited on a website in an interval can be thought of a Poisson process. 
Poisson distribution is described in terms of the rate (μ) at which the events happen. 
An event can occur 0, 1, 2, ... times in an interval. 
The average number of events in an interval is designated λ (lambda). Lambda is the event rate, also called the rate parameter. 
The probability of observing k events in an interval is given by the equation: 
P(k events in interval) = e^(-lambda) * (lambda^k / k!)
'''
'''
poisson distribution is a limiting case of a binomial distribution under the following conditions:
- The number of trials is indefinitely large or n tends to infinity
- The probability of success for each trial is same and indefinitely small or p tends to zero
- np = lambda, is finite.
'''
rv_array = spss.poisson.rvs(mu=3, size=10000) # size specifies number of random variates, loc corresponds to mean, scale corresponds to standard deviation
sns.distplot(rv_array, kde=False)


# Binomial Distribution
# distribution where only two outcomes are possible, such as success or failure, gain or loss, win or lose 
# and where the probability of success and failure is same for all the trials. 
# The outcomes need not be equally likely, and each trial is independent of each other.
'''
f(k;n,p) = nCk * (p^k) * ((1-p)^(n-k))
nCk = (n)! / ((k)! * (n-k)!) 
n=total number of trials
p=probability of success in each trial
'''
rv_array = spss.binom.rvs(n=10,p=0.8,size=10000) # n = number of trials, p = probability of success, size = number of times to repeat the trials
sns.distplot(rv_array, kde=False)

# Bernoulli Distribution is a special case of the binomial distribution where a single trial is conducted n=1
'''
Bernoulli distribution has only two possible outcomes, 1 (success) and 0 (failure), 
and a single trial, for example, a coin toss. 
So the random variable X which has a Bernoulli distribution can take value 1 with the probability of success, p, 
and the value 0 with the probability of failure, q or 1-p. 
The probabilities of success and failure need not be equally likely. 
Probability mass function of Bernoulli distribution:
f(k;p) = (p^k) * ((1-p)^(1-k))

Bernoulli distribution is a special case of the binomial distribution where a single trial is conducted (n=1)
'''
rv_array = spss.bernoulli.rvs(size=10000,p=0.6) # p = probability of success, size = number of times to repeat the trial
sns.distplot(rv_array, kde=False)


