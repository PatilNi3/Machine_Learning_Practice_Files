# OUTLIER DETECTION
'''
- Outliers are generally defined as samples that are exceptionally far from the mainstream of data.
- An outlier may also be explained as a piece of data or observation that deviates drastically from the given norm or
  average of the data set.
- Outlier Detection may be defined as the process of detecting and subsequently excluding outliers from a given set of data.
'''

# OUTLIER DETECTION TECHNIQUE:
'''
1. Scatter Plot
2. Z-Score
3. IQR (Inter Quartile Range)
4. Box Plot
'''
import pandas as pd
import seaborn as sns
import matplotlib .pyplot as plt
# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# 1. SCATTER PLOT
'''
- If there is a regression line on a scatter plot, you can identify outliers. An outlier for a scatter plot is the point 
  or points that are farthest from the regression line. There is at least one outlier on a scatter plot in most cases, 
  and there is usually only one outlier.
'''

# EXAMPLE of SCATTER PLOT
'''
data = pd.read_csv('Automobile_data.csv')
df = pd.DataFrame(data)
R = sns.lmplot(x="city-mpg", y="price", data=df)
plt.show()
R.savefig('regplot.png')
'''

# EXPLANATION:
'''
- Distance from a point to the regression line is the length of the line segment that is perpendicular to the regression 
  line and extends from the point to the regression line.
- If one point of a scatter plot is farther from the regression line than some other point, then the scatter plot has 
  at least one outlier. If a number of points are the same farthest distance from the regression line, then all these 
  points are outliers.
- If all points of the scatter plot are the same distance from the regression line, then there is no outlier.
- Refer image -> https://www2.southeastern.edu/Academics/Faculty/dgurney/Outlier.jpg
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 2. Z-SCORE
'''
- The Z-score (also called the standard score) is an important concept in statistics that indicates how far away a 
  certain point is from the mean.
- If the z score of a data point is more than 3, it indicates that the data point is quite different from the other 
  data points. Such a data point can be an outlier.
- For example - A z-score of 2 would mean tha data point is 2 standard deviation away from the mean.
- Z-score = { [Xi - mean] / standard deviation } 
'''

# EXAMPLE of Z-SCORE
'''
data = pd.read_csv('AgeData.csv')

df = pd.DataFrame(data)
print(df)

df['Z-score'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()
print(df['Z-score'])
print(df)

Final_Dataset = df[~((df['Z-score'] > 2) | (df['Z-score'] < -2))]
print(Final_Dataset)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 3. IQR (INTER QUARTILE RANGE)
'''
- Inter Quartile Range (IQR) is one of the most extensively used procedure for outlier detection and removal. 
  According to this procedure, we need to follow the following steps:

  1. Find the first quartile, Q1.
  2. Find the third quartile, Q3.
  3. Calculate the IQR, IQR = Q3-Q1.
  4. Define the normal data range with lower limit as Q1–1.5*IQR and upper limit as Q3+1.5*IQR.

- Any data point outside this range is considered as outlier and should be removed for further analysis.
'''

# EXAMPLE of IQR #
'''
data = pd.read_csv('AgeData.csv')

df = pd.DataFrame(data)

Q1 = df['Age'].quantile(0.25)
print(Q1)

Q3 = df['Age'].quantile(0.75)
print(Q3)

# We know
IQR = Q3 - Q1
print(IQR)

Lower_Limit = Q1 - (1.5 * IQR)
print(Lower_Limit)

Upper_Limit = Q3 + (1.5 * IQR)
print(Upper_Limit)

Outlier_Age = df[((df['Age'] > Upper_Limit) | (df['Age'] < Lower_Limit))]
print(Outlier_Age)

Final_Dataset = df[~((df['Age'] > Upper_Limit) | (df['Age'] < Lower_Limit))]
print(Final_Dataset)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 4. BOX PLOT-
'''
- A box plot is generally used to summerize the distribution of data sample.
- The x-axis used to represent the data sample and y-axis is used to represent the observation values.
- A box is used to summerize the starting 25 percentile, middle 50 percentile, ending 75 percentile.
- Lines called whiskers, which is extended from both the ends of box calculated as 1.5 * IQR to demonstrate the expected
  range of sensible values in the distribution.
- Observation outside the whiskers is outliers.
- We use sns.boxplot(x=" ", y=" ", data=df) in the code to plot box and whisker plot.
'''

# EXAMPLE of BOX PLOT:
'''
list = [1, 2, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 24, 25]

df = pd.DataFrame(list)

sns.boxplot(data=df)
plt.show()
'''




# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻