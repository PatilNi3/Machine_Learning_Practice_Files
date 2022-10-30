import pandas as pd

# WHAT IS DATA ?
'''
- Data a can be defined as a systematic record of a particular quantity.
- Data is raw, unorganized facts that need to be processed. Data can be something simple and seemingly random and
  useless until it is organized.
'''
# TYPES OF DATA:
'''
1. Categorical (Qualitative)
   a. Nominal
   b. Ordinal
2. Numerical (Quantitative)
   a. Discrete
   b. Contineous
'''

# 1. CATEGORICAL:
'''
- Qualitative data is used to represent some characteristics or attributes of the data.
- These are non-numerical in nature.
- Examples:
    a. Attitudes of people to a political system.
    b. Music and art
    c. Intelligence
    d. Beauty of a person
'''

# 1.1 Nominal:
'''
- Nominal data attributes can’t either be ordered or measured.
- Nominal data attributes are letters, symbols or gender, etc.
- Examples:
    a. Gender (Women, Men)
    b. Eye color (Blue, Green, Brown)
    c. Hair color (Blonde, Brown, Brunette, Red, etc.)
    d. Marital status (Married, Single)
    e. Religion (Muslim, Hindu, Christian)
'''

# 1.2 Ordinal:
'''
- Ordinal data is the specific type of data that follows a natural order.  
- The difference between the data values is not determined in the case of nominal data. 
- Ordinal data is mostly found in surveys, economics, questionnaires, and finance operations.
- Examples:
    a. Feedback is recorded in the form of ratings from 1-10.
    b. Education level: elementary school, high school, college.
    c. Economic status: low, medium, and high.
    d. Letter grades: A, B, C, and etc.
    e. Customer level of satisfaction: very satisfied, satisfied, neutral, dissatisfied, very dissatisfied.
'''

# 2. NUMERICAL:
'''
- Quantitative data can be measured and is not just observable.
- Numerical data is indicated by quantitative data.
- Examples:
    a. Daily temperature
    b. Price
    c. Weights
    d. Income
'''

# 2.1 Discrete:
'''
- Discrete data refers to the data values which can only attain certain specific values. 
- Discrete data can’t attain a range of values.
- Examples:
    a. The number of students in a class,
    b. The number of chips in a bag,
    c. The number of stars in the sky
'''

# 2.2 Contineous:
'''
- Continuous Data can contain values between a certain range that is within the highest and lowest values.
- For example, the heights of the students in the class can be largely varying in nature, therefore, they can be 
  divided into ranges to summarise the data.
- Examples:
    a. Height and weight of a student,
    b. Daily temperature recordings of a place
    c. Wind speed measurement
'''

# ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ ENCODING ☺ ☺ #

# WHAT IS CATEGORICAL ENCODING?
'''
- Typically, any structured dataset includes multiple columns – a combination of numerical as well as categorical variables. 
- A machine can only understand the numbers. It cannot understand the text. That’s essentially the case with Machine 
  Learning algorithms too.
- That’s primarily the reason we need to convert categorical columns to numerical columns so that a machine learning 
  algorithm understands it. This process is called categorical encoding.
- Categorical encoding is a process of converting categories to numbers.
'''

# TYPES OF ENCODING:

# 1. ONE-HOT ENCODING:
'''
- In one-hot encoding, we create a new set of dummy (binary) variables that is equal to the number of categories (k) in 
  the variable. 
- For example, let’s say we have a categorical variable Color with three categories called “Red”, “Green” and “Blue”, we 
  need to use three dummy variables to encode this variable using one-hot encoding. 
- A dummy (binary) variable just takes the value 0 or 1 to indicate the exclusion or inclusion of a category.
'''

# EXAMPLE:
'''
data = pd.read_csv('OneHotEncoding.csv')
df = pd.DataFrame(data)
print(df)
'''

# Explanation:
'''
In above One Hot Encoding example,
  “Red” color is encoded as [1 0 0]
  “Green” color is encoded as [0 1 0]
  “Blue” color is encoded as [0 0 1]
'''

# Dummy Variable Trap (associated with one hot encoding)
'''
- Dummy Variable Trap is a scenario in which variables are highly correlated to each other.
- The Dummy Variable Trap leads to the problem known as multicollinearity. Multicollinearity occurs where there is a 
  dependency between the independent features. Multicollinearity is a serious issue in machine learning models like 
  Linear Regression and Logistic Regression.
'''

# 2. DUMMY ENCODING:
'''
- Dummy encoding also uses dummy (binary) variables. Instead of creating a number of dummy variables that is equal to the 
  number of categories (k) in the variable, dummy encoding uses k-1 dummy variables. 
- To encode the same Color variable with three categories using the dummy encoding, we need to use only two dummy variables.
'''

# EXAMPLE:
'''
# data = pd.read_csv('DummyEncoding.csv')
# df = pd.DataFrame(data)
# print(df)
'''

# Explanation:
'''
In above Dummy Encoding example,
  “Red” color is encoded as [1 0]
  “Green” color is encoded as [0 1]
  “Blue” color is encoded as [0 0]

Dummy encoding removes a duplicate category present in the one-hot encoding.
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# EXAMPLE:
"""
# 1. Reading Dataset

data = pd.read_csv("diamonds.csv")

df = pd.DataFrame(data)

# 2. Checking for Null values

a = df.isnull().sum()
print(a)

# 3. Information of Dataset to check datatype

b = df.info()
print(b)                        # There are 3 categorical variables in the dataset. They are cut, color and clarity.

# 4. Let’s see the unique categories or labels of these 3 variables.

CUT = df['cut'].unique()
print(CUT)
COLOR = df['color'].unique()
print(COLOR)
CLARITY = df['clarity'].unique()
print(CLARITY)

'''
- There are 5, 7, 8 unique categories in the cut, color, clarity variables respectively.
- To encode this variable, we need to create 5, 7, 8 dummy variables respectively in one-hot encoding and 4, 6, 7 dummy 
  variables in dummy encoding respectively.
'''
"""

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# IMPLEMENTATION OF ONE-HOT ENCODING USING PANDAS (to all categorical variables in the dataset)
"""

data = pd.read_csv("diamonds.csv")
df = pd.DataFrame(data)

one_hot_df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=False)
print(one_hot_df)
one_hot_df.to_csv('OneHot.csv')

'''
- The encoded dataset has 27 variables. This is because one-hot encoding has added 20 extra dummy variables when encoding 
  the categorical variables. 
- So, one-hot encoding expands the feature space (dimensionality) in your dataset.
'''

"""

# ---------------------------------------------------------------------------------------------------------------------#

# IMPLEMENTATION OF DUMMY ENCODING USING PANDAS (to all categorical variables in the dataset)
"""

data = pd.read_csv("diamonds.csv")
df = pd.DataFrame(data)

dummy_df = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
print(dummy_df)                 # Fair, D, I1
dummy_df.to_csv('Dummy.csv')

'''
- The encoded dataset has 24 variables. This is because dummy encoding has added 17 extra dummy variables when encoding 
  the categorical variables. 
- So, dummy encoding also expands the feature space (dimensionality) in your dataset.
'''

"""
# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# IMPLEMENTATION OF ONE-HOT ENCODING USING SCIKIT LEARN

'''
data = pd.read_csv("diamonds.csv")
df = pd.DataFrame(data)

from sklearn.preprocessing import OneHotEncoder

One_Hot_Encoding = OneHotEncoder(categories='auto', drop=None, sparse=False)

encoded_data = One_Hot_Encoding.fit_transform(df[['cut', 'color', 'clarity']])
print(encoded_data)
print(encoded_data.shape)

# OneHot_Df = pd.DataFrame(One_Hot_Encoding.fit_transform(df[['cut', 'color', 'clarity']]))
# print(OneHot_Df)
'''

# ----------------------------------------------------------------------------------------------------------------------#

# IMPLEMENTATION OF DUMMY ENCODING USING SCIKIT LEARN

'''
data = pd.read_csv("diamonds.csv")
df = pd.DataFrame(data)

from sklearn.preprocessing import OneHotEncoder

Dummy_Encoding = OneHotEncoder(categories='auto', drop='first', sparse=False)

encoded_data = Dummy_Encoding.fit_transform(df[['cut', 'color', 'clarity']])
print(encoded_data)
print(encoded_data.shape)

# Dummy_Df = pd.DataFrame(Dummy_Encoding.fit_transform(df[['cut', 'color', 'clarity']]))
# print(Dummy_Df)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 3. ORDINAL:
'''
In ordinal Logistic regression, there can be 3 or more possible ordered types of dependent variables, such as "low", 
"Medium", or "High".
'''

'''☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻'''