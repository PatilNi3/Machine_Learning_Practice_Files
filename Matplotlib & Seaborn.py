import pandas as pd
# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# MATPLOTLIB #
'''
- Matplotlib is the most popular data visualization package in Python.
- Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

- Installing Matplotlib -> pip install matplotlib
- Importing Matplotlib -> import matplotlib.pyplot as plt   or   from matplotlib import pyplot as plt
'''
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------#

# SEABORN #
'''
- Seaborn is a Python data visualization library based on matplotlib. 
- It will be used to visualize random distributions.
- Seaborn comes with built in themes for styling Matplotlib graphics.

- Installing Seaborn -> pip install seaborn
- Importing Seaborn -> import seaborn as sns
'''
import seaborn as sns

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# WHAT IS DATA VISUALIZATION ?
'''
- Data visualization is defined as a graphical representation that contains the information and the data.
- By using visual elements like charts, graphs, and maps, data visualization techniques provide an accessible way to see 
  and understand trends, outliers, and patterns in data.
'''

# Classification of analysis for Data Visualisation
'''
1. Univariate Analysis: This type of data involves only one variables.

    a. Distribution Plot -> Library: SeaBorn -> Syntax: sns.FacetGrid(df, hue=" ", size= )
    b. Box Plot -> Library: SeaBorn -> Syntax: sns.boxplot(x=" ",y=" ", data=df)
    c. Violin Plot -> Library: SeaBorn -> Syntax: sns.violinplot(x=" ",y=" ", data=df, size= )
    d. Pair Plot -> Library: Seaborn -> Syntax: sns.pairplot(df, hue=" ", palette=" ")
    e. Join Plot -> Lirary: Seaborn -> Syntax: sns.jointplot(x ='total_bill', y ='tip', data = df)
    f. Heat Maps -> Library: seaborn -> syntax: sns.heatmap(df)

2. Bivariate Analysis: This type of data involves two different variables.

    a. Line Plot -> Library: MatPlotLib -> Syntax: plt.plot(x, y)
    b. Bar Plot -> Library: MatPlotLib -> Syntax: plt.bar(x, y)
    c. Scatter Plot -> Library: MatPlotLib -> Syntax: plt.scatter(x, y)
    d. Pie Chart -> Library: MatPlotLib -> Syntax: df.plot.pie()
    e. Histogram Plot -> Library: MatPlotLib -> Syntax: plt.hist()
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 1. DISTRIBUTION PLOT:
'''
- Distribution plot takes as input an array and plots a curve corresponding to the distribution of points in the array.

- We can compare the distribution plot in Seaborn to histograms in Matplotlib. 
  They both offer pretty similar functionalities. Instead of frequency plots in the histogram, here we’ll plot an 
  approximate probability density across the y-axis.

- We using sns.distplot() in the code to plot distribution graphs.

# Here, the curve(KDE) that appears drawn over the distribution graph is the approximate probability density curve.
# KDE stands for Kernal Density Estimation
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
Age = [26, 27, 30, 24, 27]

sns.distplot(Age)
plt.show()
'''

# EXAMPLE: 2
'''
Age = [26, 27, 30, 24, 27]

sns.distplot(Age, vertical=True)                      # kuch samajh me nahi aa raha
plt.show()
'''

# EXAMPLE: 3
'''
Age = [26, 27, 30, 24, 27]

sns.distplot(Age, kde=False)            # KDE stands for Kernal Density Estimation it means a curve
plt.show()
'''

# EXAMPLE: 4
'''
Age = [26, 27, 30, 24, 27]

sns.distplot(Age, kde=False, color='red')       # adding colour to the graph
plt.show()
'''

# EXAMPLE: 5
'''
Age = [26, 27, 30, 24, 27]

sns.distplot(Age, bins=20)                      # kuch samajh me nahi aa raha
plt.show()
'''

# EXAMPLE: 6
'''
data = pd.read_csv("Automobile_data.csv")

Auto = data['price']
sns.distplot(Auto)
plt.show()

# or

sns.distplot(data.price)                    # kuch samajh me nahi aa raha
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 2. BOX and WHISKER PLOT:
'''
- A box plot is generally used to summerize the distribution of data sample.
- The x-axis used to represent the data sample and y-axis is used to represent the observation values.
- A box is used to summerize the starting 25 percentile, middle 50 percentile, ending 75 percentile
- Lines called whiskers, which is extended from both the ends of box calculated as 1.5 * IQR to demonstrate the expected
  range of sensible values in the distribution.
- Observation outside the whiskers is outliers. 
- We use sns.boxplot(x=" ", y=" ", data=df) in the code to plot box and whisker plot.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = pd.read_csv("table.csv")

sns.boxplot(x='age-months', y='Speed-m/sec', data=df)
plt.show()
'''

# EXAMPLE: 2
'''
df = pd.read_csv("table.csv")

sns.boxplot(x='height-inches', y='weight-lbs', hue='gender', data=df)
plt.show()
'''

# EXAMPLE: 3
'''
df = pd.read_csv("table.csv")

sns.boxplot(x='gender', y='weight-lbs', data=df, order=['M', 'F'])
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 3. VIOLIN PLOT:
'''
- The violin plots can be considered as a combination of Box plot at the middle and distribution plots (Kernel Density 
  Estimation) on both sides of the data. It shows the entire distribution of data.
- Median (a white dot on the violin plot), 
  Interquartile range (the black bar in the center of violin)
  The lower/upper adjacent values (the black lines stretched from the bar) is defined as 
  "first quartile — 1.5  * IQR" and 
  "third quartile + 1.5 * IQR" respectively. 
  These values can be used in a simple outlier detection technique (Tukey’s fences); observations lying outside of 
  these “fences” can be considered outliers.
- We use sns.violinplot(x=" ", y=" ", data=df, size= ) in the code to plot the violin plot.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = pd.read_csv('table.csv')

Vio = sns.violinplot(x='gender', y='age-months', data=df)                                # x or y, one should be numeric
plt.show()
'''

# EXAMPLE: 2
'''
df = pd.read_csv('table.csv')

Vio = sns.violinplot(x='gender', y='age-months', data=df, inner='quartile')              # x or y, one should be numeric
plt.show()
'''

# EXAMPLE: 3
'''
df = pd.read_csv('table.csv')

Vio = sns.violinplot(x='age-months', y='Group', hue='gender', data=df)                   # x or y, one should be numeric
plt.show()
'''

# EXAMPLE: 4
'''
df = pd.read_csv('table.csv')

Vio = sns.violinplot(x='age-months', y='Group', hue='gender', split=True, data=df)       # x or y, one should be numeric
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 4. PAIR PLOT:
'''
- Pair Plots are a really simple (one-line-of-code simple!) way to visualize relationships between each variable.
- It produces a matrix of relationships between each variable in your data.
- A pairs plot is a matrix of scatterplots that lets you understand the pairwise relationship between different 
  variables in a dataset.
- We use sns.pairplot(x=" ", y=" ", data=df) in the code to plot the pairplot.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = pd.read_csv('table.csv')

P = sns.pairplot(df, vars= ['age-months', 'height-inches', 'weight-lbs', 'leg-length-inches', 'Speed-m/sec'])
plt.show()
P.savefig('pairplot.png')               # when output is not showing
'''

# EXAMPLE: 2
'''
df = pd.read_csv('table.csv')

P = sns.pairplot(df, vars= ['age-months', 'height-inches', 'weight-lbs', 'leg-length-inches', 'Speed-m/sec'], hue='gender')
plt.show()
P.savefig('pairplot1.png')
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 5. JOIN PLOT:
'''
- It is used to draw a plot of two variables with bivariate and univariate graphs. It basically combines two different 
  plots.
- sns.jointplot(x ='total_bill', y ='tip', data = df)
- sns.jointplot(x ='total_bill', y ='tip', data = df, kind ='kde')
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = pd.read_csv('table.csv')

sns.jointplot(x='age-months', y='height-inches', data=df)               # by default considering histogram
plt.show()
'''

# EXAMPLE: 2
'''
df = pd.read_csv('table.csv')

sns.jointplot(x='age-months', y='height-inches', hue='gender', data=df)
plt.show()
'''

# EXAMPLE: 3
'''
df = pd.read_csv('table.csv')

sns.jointplot(x='age-months', y='height-inches', hue='gender', kind='kde', data=df)
plt.show()
'''

# EXAMPLE: 4
'''
df = pd.read_csv('table.csv')

sns.jointplot(x='age-months', y='height-inches', hue='gender', marker="+", data=df)
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 6. HEAT MAP:
'''
- Heatmap is defined as a graphical representation of data using colors to visualize the value of the matrix. A heatmap 
  is a plot of rectangular data as a color-encoded matrix. As parameter it takes a 2D dataset.
- Heat maps can help the user visualize simple or complex information.
- Heatmaps in Seaborn can be plotted by using the seaborn.heatmap(df) function.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = [[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
      [10, 25, 50, 75, 100, 20, 40, 60, 80, 0],
      [27, 24, 7, 22, 25, 17, 11, 96, 95, 35]]

sns.heatmap(df, annot= True)
plt.show()
'''

# EXAMPLE: 2
'''
df = pd.read_csv('table.csv')

df.drop(columns=['gender', 'Group'], inplace=True)

plt.figure(figsize=(30,10))

sns.heatmap(df)
plt.show()
'''

# EXAMPLE: 3
'''
df = pd.read_csv('table.csv')

df.drop(columns=['gender', 'Group'], inplace=True)

plt.figure(figsize=(30,10))

sns.heatmap(df, cmap='coolwarm')
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 6.1 HEATMAP CORRELATION
'''
- A correlation heatmap is a graphical representation of correlation matrix representing the "correlaton between 
  diffrent variables."
- The value of correlation can take any value from -1 to 1.
- Correlation between two variables can also be determined using a scatter plot between these two variables.
- Heatmap correlation can be ploted by using the sns.heatmap(df.corr()) function.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
df = pd.read_csv('table.csv')

df.drop(columns=['gender', 'Group'], inplace=True)

plt.figure(figsize=(30,10))

sns.heatmap(df.corr())
plt.show()
'''

# EXAMPLE: 2
'''
df = pd.read_csv('table.csv')

df.drop(columns=['gender', 'Group'], inplace=True)

plt.figure(figsize=(30,10))

sns.heatmap(df.corr(), cmap='coolwarm')
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 1. LINE PLOT:
'''
- Line charts are used to represent the relation between two data, numerical values on one axis and categorical values 
  on other axis.
- We using "plt.plot(x, y)" in the code to plot line plot and "plt.show()" to view plot
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.plot(x, y)
plt.show()
'''

# EXAMPLE: 2
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.plot(x, y, linestyle='-')
plt.show()

plt.plot(x, y, linestyle='--')
plt.show()

plt.plot(x, y, linestyle='-.')
plt.show()

plt.plot(x, y, linestyle=':')
plt.show()
'''

# EXAMPLE: 3
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.plot(x, y, linestyle='-.', linewidth=5)
plt.show()
'''

# EXAMPLE: 4
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.plot(x, y, color='red', linestyle='-.', linewidth='5')
plt.show()
'''

# EXAMPLE: 5
'''
x = [2, 20]
y = [5, 5]

plt.plot(x, y, 'y')
plt.show()
'''

# EXAMPLE: 6
'''
a = [5, 50]
b = [3, 3]
c = [8, 8]

# Plot a horizontal line
plt.plot(a, b, '-r', linewidth=3, label='red line')
plt.plot(a, c, '-b', linewidth=3, label='blue line')
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 2. BAR PLOT:
'''
- A bar plot or bar chart is a graph that represents the category of data with rectangular bars with lengths and heights 
  that is proportional to the values which they represent. 
- The bar plots can be plotted horizontally or vertically. 
- A bar chart describes the comparisons between the discrete categories. 
- One of the axis of the plot represents the specific categories being compared, while the other axis represents the 
  measured values corresponding to those categories.
- We use plt.bar(df, height, width, bottom, align) in the code to plot the barplot.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.bar(x, y)
plt.show()

plt.bar(x, y, color='black')
plt.show()

plt.bar(x, y, color='orange', width=2)
plt.show()
'''

# EXAMPLE: 2
'''
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

plt.bar(x, y)
plt.xlabel('Table of 2')
plt.ylabel('Table of 5')
plt.title('Comparing tables of 2 and 5')
plt.show()
'''

# EXAMPLE: 3
'''
a = ['Nitin', 'Neha', 'Rucha', 'Pratima', 'Ankita']
b = [26, 27, 30, 24, 27]

plt.bar(a, b, color='orange')
plt.xlabel('Students', fontweight='bold')
plt.ylabel('Age', fontweight='bold')
plt.title('Age graph', fontweight='bold')
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 3. SCATTER PLOT:
'''
- The coordinates of each point are defined by two dataframe columns and filled circles are used to represent each point. 
- This kind of plot is useful to see complex correlations between two variables.
- Scatter Plot is also used to find the outliers from datasets.
- We using "df.plot.scatter(x=" ", y=" ", c="select colour")" to plot the scatter plot.
- There is 3 types of scatter plots:
  1. Positive - A scatter plot with increasing values of both variables can be said to have a positive correlation.
  2. Negative - A scatter plot with an increasing value of one variable and a decreasing value for another variable
  3. No relation - A scatter plot with no clear increasing or decreasing trend in the values of the variables
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
DATA = {'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
       'Age': [26, 27, 30, 24, 27, 25, 31, 26, 28, 30, 22, 20, 24, 21, 23, 25, 27, 29, 31, 18, 20, 19, 17, 15, 32, 18]}

df = pd.DataFrame(data=DATA)

df.plot.scatter(x='Name', y='Age', color='green')
plt.show()
'''

# EXAMPLE: 2
'''
DATA = {'Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
       'Age': [26, 27, 30, 24, 27, 25, 31, 26, 28, 30, 22, 20, 24, 21, 23, 25, 27, 29, 31, 18, 20, 19, 17, 15, 32, 18]}

df = pd.DataFrame(data=DATA)

df.plot.scatter(x='Name', y='Age', s=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26], c='green')     # s=size of dots
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 4. PIE CHART:
'''
- Pie Chart is generally used to analyze the data on how a numeric variable changes across different categories.
- The Python data visualization library Seaborn doesn’t have a default function to create pie charts, but we can use 
  in Matplotlib to create a pie chart and add a Seaborn color palette.
- we use plt.pie(" ", labels=" ")
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
languages = ['Python', 'Java', 'JavaScript', 'C#', 'PHP', 'C & C++', 'other']
percentage = [25.95, 21.42, 8.26, 7.62, 7.37, 6.31, 23.07]

plt.pie(percentage, labels=languages)
plt.show()
'''

# EXAMPLE: 2
'''
Name = ['Nitin', 'Neha', 'Rucha', 'Pratima', 'Ankita']
Age = [26, 27, 30, 24, 27]

plt.pie(Age, labels=Name)
plt.show()
'''

# ----------------------------------------------------------------------------------------------------------------------#
# 5. HISTOGRAM PLOT:
'''
- A histogram is a graph showing frequency distributions.
- It is a graph showing the number of observations within each given interval.
- It is accurate method for the graphical representation of numerical data distribution.
- It is a type of bar plot where X-axis represents the bin ranges while Y-axis gives information about frequency.
- We use the plt.hist() function to create histograms.
'''
# ----------------------------------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
Age = [26, 27, 30, 24, 27, 26]

# Matplotlib
plt.hist(Age)
plt.show()

# Seaborn
sns.histplot(Age, kde=False)
plt.show()
'''

# EXAMPLE: 2
'''
Salary = [1200000, 1500000, 1800000, 2200000, 2500000, 2800000, 3200000, 3500000, 2500000, 3500000, 2200000, 2500000,
          2800000, 1500000, 1800000, 2200000, 2500000, 1500000, 1800000, 2200000, 2500000, 2800000, 3200000, 3500000,
          1200000, 1500000, 1800000, 2200000, 2500000, 2800000, 3200000, 3500000, 2500000, 3500000, 2200000, 2500000,
          2200000, 2500000, 2800000, 3200000, 3500000, 2500000, 2500000, 2800000, 3200000, 3500000, 1800000, 2200000]

# Matplotlib
plt.hist(Salary)
plt.show()

# Seaborn
sns.histplot(Salary, kde=False)
plt.show()
'''



# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻