



from scipy.stats.morestats import _calc_uniform_order_statistic_medians


Null: Given two sample means are equal
Alternate: Given two sample means are not equal

if test statistic is found to be lower than the critical value (i.e. test statistic does not fall in the critical region) then accept the null hypothesis
if test statistic is found to be greater than the critical value the null hypothesis is rejected
(theoretically null hypothesis is rejected if the test statistic falls in the critical region)
critical values are the boundaries of the critical region
critical value is a point beyond which we reject the null hypothesis

If the test is one-sided (like a χ2 test or a one-sided t-test) then there will be just one critical value, 
but in other cases (like a two-sided t-test) there will be two

The general critical value for a two-tailed test is 1.96, which is based on the fact that 
95% of the area of a normal distribution is within 1.96 standard deviations of the mean


#######################

Understanding p-value
p value of 0.0254 is 2.54% means there is a 2.54% chance your results could be random 
(i.e. happened by chance). That’s pretty tiny. 
p-value of .9(90%) means your results have a 90% probability of being completely random and not due to anything in your experiment. 
Therefore, the smaller the p-value, the more important (“significant“) your results.

p-value <= 0.05 is statistically significant - reject the null hypothesis
p-value > 0.05 is not statistically significant - accept the null hypothesis

#########################################################################################

Z-test
sample is assumed to be normally distributed

Null: Sample mean is same as the population mean
Alternate: Sample mean is not same as the population mean
z = (x — μ) / (σ / √n), where
x= sample mean
μ = population mean
σ / √n = population standard deviation

#########################################################################################

T-test - used when population parameters are not known
sample is assumed to be normally distributed

t-statistic = (x1 — x2) / (σ / √n1 + σ / √n2), where
x1 = mean of sample 1
x2 = mean of sample 2
n1 = size of sample 1
n2 = size of sample 2

There are three versions of t-test
#########################################################################################
- One sample t-test which tests the mean of a single group against a known mean.
used to compare the sample mean with the specific value (usually hypothesized population mean)

assumptions:
- sample is assumed to be normally distributed. Shapiro-Wilks Test
- Observations are independent of each other

Hypotheses:
- Null hypothesis: Sample mean is equal to the hypothesized population mean
- Alternative hypothesis: Sample mean is not equal to the hypothesized population mean (two-tailed or two-sided)
- Alternative hypothesis: Sample mean is either greater or lesser to the hypothesized population mean (one-tailed or one-sided)

Interpretation example:
p value obtained from the one sample t-test (0.35727) is not significant (as it is > 0.05)
accept the null hypothesis of equality 

#########################################################################################
- unpaired or Independent samples t-test which compares mean for two groups
Two sample t-test is used to compare the mean of two independent groups

Assumptions:
- Observations in two groups have an approximately normal distribution (Shapiro-Wilks Test)
- Homogeneity of variances (variances are equal between treatment groups) (Levene or Bartlett Test)
- The two groups are sampled independently from each other from the same population

Hypotheses:
- Null hypothesis: Two group means are equal
- Alternative hypothesis: Two group means are different (two-tailed or two-sided)
- Alternative hypothesis: Mean of one group either greater or lesser than another group (one-tailed or one-sided)

Welch's t-test performs better than Student's t-test whenever sample sizes and variances are unequal between groups, and gives the same result when sample sizes and variances are equal. 

#########################################################################################
- Paired sample t-test which compares means from the same group at different times
- For example, we have plant variety A and would like to compare the yield of A before and after the application of some fertilizer
- Note: Paired t-test is a one sample t-test on the differences between the two dependent variables

Hypotheses
- Null hypothesis: There is no difference between the two dependent variables (difference=0)
- Alternative hypothesis: There is a difference between the two dependent variables (two-tailed or two-sided)
- Alternative hypothesis: Difference between two response variables either greater or lesser than zero (one-tailed or one-sided)

Assumptions
- Differences between the two dependent variables follows an approximately normal distribution (Shapiro-Wilks Test)
- Independent variable should have a pair of dependent variables
- Differences between the two dependent variables should not have outliers
- Observations are sampled independently from each other


#########################################################################################

ANOVA, also known as analysis of variance, 
is used to compare multiple (three or more) samples with a single test

There are 2 major flavors of ANOVA
- One-way ANOVA: It is used to compare the difference between the three or more 
samples/groups of a single independent variable.
- MANOVA: It allows us to test the effect of one or more independent variable on 
two or more dependent variables. In addition, MANOVA can also detect the difference 
in co-relation between dependent variables given the groups of independent variables.

Null: All pairs of samples are same i.e. all sample means are equal
Alternate: At least one pair of samples is significantly different

The statistics used to measure the significance, in this case, is called F-statistic
F-statistic = ((SSE1 — SSE2)/m)/ SSE2/n-k, where
SSE = residual sum of squares
m = number of restrictions
k = number of independent variables

#########################################################################################

Pearsons Chi-square test is used to compare categorical variables. There are two type of chi-square test
- Goodness of fit test, which determines if a sample matches the population.
- A chi-square fit test for two independent variables is used to compare two variables in a contingency table to check if the data fits.

The hypothesis being tested for chi-square is
Null: Variable A and Variable B are independent; i.e. there is no relation between the variables
Alternate: Variable A and Variable B are not independent; i.e. there is a significant relation between the two

chi-square statistic =  Σ [ (Or,c — Er,c)2 / Er,c ] where
Or,c = observed frequency count at level r of Variable A and level c of Variable B
Er,c = expected frequency count at level r of Variable A and level c of Variable B

How to interpret the statistic 
a. A small chi-square value means that data fits
b. A high chi-square value means that data doesn’t fit.

# EXAMPLE 1
import pandas as pd
import scipy.stats as spss 
import seaborn as sns
# defining the table 
data = [[207, 282, 241], [234, 242, 232]] # this data is already in crosstab format
stat, p, dof, expected = spss.chi2_contingency(data) 
# interpret p-value 
alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 

# EXAMPLE 2
df = pd.DataFrame({'Gender' : ['M', 'M', 'M', 'F', 'F'] * 10,
                   'Smoker' : ['Y', 'Y', 'N', 'N', 'Y'] * 10 })
# To run the Chi-Square Test, the easiest way is to convert the data into a 
# contingency table with frequencies. We will use the crosstab command from pandas.
contigency = pd.crosstab(df.Gender, df.Smoker) 
contigency_pct = pd.crosstab(df.Gender, df.Smoker, normalize='index') # If we want the percentages by column, then we should write normalize=’column’ and if we want the total percentage then we should write normalize=’all’
# easy way to see visually the contingency tables are the heatmaps
sns.heatmap(contigency, annot=True, cmap="YlGnBu")
# Chi square test
stat, p, dof, expected = spss.chi2_contingency(contigency) 
# interpret p-value 
alpha = 0.05
print("p value is " + str(p)) 
if p <= alpha: 
    print('Dependent (reject H0)') 
else: 
    print('Independent (H0 holds true)') 

#########################################################################################

'''
Central Limit Theorem

The central limit theorem states that the sampling distribution of the mean of any independent, random variable will be normal or nearly normal, if the sample size is large enough.

How large is "large enough"? The answer depends on two factors.

    Requirements for accuracy. The more closely the sampling distribution needs to resemble a normal distribution, the more sample points will be required.
    The shape of the underlying population. The more closely the original population resembles a normal distribution, the fewer sample points will be required.

In practice, some statisticians say that a sample size of 30 is large enough when the population distribution is roughly bell-shaped. Others recommend a sample size of at least 40. But if the original population is distinctly not normal (e.g., is badly skewed, has multiple peaks, and/or has outliers), researchers like the sample size to be even larger.
'''


'''
z-score (aka, a standard score) indicates how many standard deviations an element is from the mean
z = ( value of element - population mean ) / population sd
Here is how to interpret z-scores.

    A z-score less than 0 represents an element less than the mean.
    A z-score greater than 0 represents an element greater than the mean.
    A z-score equal to 0 represents an element equal to the mean.
    A z-score equal to 1 represents an element that is 1 standard deviation greater than the mean; a z-score equal to 2, 2 standard deviations greater than the mean; etc.
    A z-score equal to -1 represents an element that is 1 standard deviation less than the mean; a z-score equal to -2, 2 standard deviations less than the mean; etc.
    If the number of elements in the set is large, about 68% of the elements have a z-score between -1 and 1; about 95% have a z-score between -2 and 2; and about 99% have a z-score between -3 and 3.
'''
'''
Molly earned a score of 940 on a national achievement test. The mean test score was 850 with a standard deviation of 100. What proportion of students had a higher score than Molly? (Assume that test scores are normally distributed.)

(A) 0.10
(B) 0.18
(C) 0.50
(D) 0.82
(E) 0.90

Solution

The correct answer is B. As part of the solution to this problem, we assume that test scores are normally distributed. In this way, we use the normal distribution to model the distribution of test scores in the real world. Given an assumption of normality, the solution involves three steps.

    First, we transform Molly's test score into a z-score, using the z-score transformation equation.

    z = (X - μ) / σ = (940 - 850) / 100 = 0.90

    Then, using an online calculator (e.g., Stat Trek's free normal distribution calculator), a handheld graphing calculator, or the standard normal distribution table, we find the cumulative probability associated with the z-score. In this case, we find P(Z < 0.90) = 0.8159.

    Therefore, the P(Z > 0.90) = 1 - P(Z < 0.90) = 1 - 0.8159 = 0.1841.

Thus, we estimate that 18.41 percent of the students tested had a higher score than Molly. 
'''


'''
The t distribution has the following properties:

    The mean of the distribution is equal to 0 .
    The variance is equal to v / ( v - 2 ), where v is the degrees of freedom (see last section) and v > 2.
    The variance is always greater than 1, although it is close to 1 when there are many degrees of freedom. With infinite degrees of freedom, the t distribution is the same as the standard normal distribution.

'''
# t statistic (also known as the t score) = [ sample mean - population mean ] / [ sample sd / sqrt( sample size ) ]

'''
degrees of freedom refers to the number of independent observations in a set of data = (sample size - 1)
'''

'''
Acme Corporation manufactures light bulbs. The CEO claims that an average Acme light bulb lasts 300 days. A researcher randomly selects 15 bulbs for testing. The sampled bulbs last an average of 290 days, with a standard deviation of 50 days. If the CEO's claim were true, what is the probability that 15 randomly selected bulbs would have an average life of no more than 290 days?

Note: There are two ways to solve this problem, using the T Distribution Calculator. Both approaches are presented below. Solution A is the traditional approach. It requires you to compute the t statistic, based on data presented in the problem description. Then, you use the T Distribution Calculator to find the probability. Solution B is easier. You simply enter the problem data into the T Distribution Calculator. The calculator computes a t statistic "behind the scenes", and displays the probability. Both approaches come up with exactly the same answer.

Solution A

The first thing we need to do is compute the t statistic, based on the following equation:

t = [ x - μ ] / [ s / sqrt( n ) ]
t = ( 290 - 300 ) / [ 50 / sqrt( 15) ]
t = -10 / 12.909945 = - 0.7745966

where x is the sample mean, μ is the population mean, s is the standard deviation of the sample, and n is the sample size.

Now, we are ready to use the T Distribution Calculator. Since we know the t statistic, we select "T score" from the Random Variable dropdown box. Then, we enter the following data:

    The degrees of freedom are equal to 15 - 1 = 14.
    The t statistic is equal to - 0.7745966.

The calculator displays the cumulative probability: 0.226. Hence, if the true bulb life were 300 days, there is a 22.6% chance that the average bulb life for 15 randomly selected bulbs would be less than or equal to 290 days.
'''

'''
Suppose scores on an IQ test are normally distributed, with a population mean of 100. Suppose 20 people are randomly selected and tested. The standard deviation in the sample group is 15. What is the probability that the average test score in the sample group will be at most 110?

Solution:

To solve this problem, we will work directly with the raw data from the problem. We will not compute the t statistic; the T Distribution Calculator will do that work for us. Since we will work with the raw data, we select "Sample mean" from the Random Variable dropdown box. Then, we enter the following data:

    The degrees of freedom are equal to 20 - 1 = 19.
    The population mean equals 100.
    The sample mean equals 110.
    The standard deviation of the sample is 15.

We enter these values into the T Distribution Calculator. The calculator displays the cumulative probability: 0.996. Hence, there is a 99.6% chance that the sample average will be no greater than 110.
'''


