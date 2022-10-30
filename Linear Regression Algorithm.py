
# WHAT IS LINEAR REGRESSION ALGORITHM ?
'''
- Linear Regression is a statistical model used to predict the relationship between independent and dependent variable.
- Linear Regression makes prediction for contineous/real or numeric variables.
- Linear Regression Algorithm shows a linear relationship between a dependent(Y) and one or more independent(X) variables,
  hence called as Linear Regression.
'''

# TYPES OF LINEAR REGRESSION

# 1. SIMPLE LINEAR REGRESSION
'''
- If a single independent variable is used to predict the value of numerical dependent variable, then such a linear 
  regression algorithms called simple linear regression.
'''

# 2. MULTIPLE LINEAR REGRESSION
'''
- If more than one independet variable is used to predict the value of a numerical dependent variable, then such a linear 
  regression algorithm is called as multiple linear regression.
'''

# WHAT IS BEST FIL LINE ?
'''
- Amongst evaluated different possible lines, the lines goes through maximum datapoints, minimises the distance between 
  other points and line, called as best fit line.
- When working with linear regression, our main goal is to find the best fit line that means the error between predicted
  values and actual values should be minimised. The best fit line have the least error.
- The different values for the coefficient of lines(m) gives a different line of regression, so we need to calculate the 
  best values for m to find the best fit line, so to calculate this we use Cost Function.
'''

# ERROR
'''
- Distance between other points and line called as Errors.
'''

# MEAN SQUARED ERROR
'''
- For linear regression we use the Mean Squared Error (MSE) Cost Function, which is the average of squared error occured
  between the predicted value and actual values.
'''

# GRADIENT DESCENT
'''
- The algorithm will find out best fit line from the infinite number of possibilities which have minimum sum of squared 
  error with the help of process called as Gradient Descent.
- Gradient descent is used to minimise the MSE by calculating the gradient of the Cost Function.
- A regression model use gradient descent to update the coefficient of the line by reducing the cost function.
- It is done by random selection of values of coefficient and then iteratively update the values to reach the minimum 
  Cost Function.
'''

# GLOBAL MINIMA
'''
- The point where function takes the minimum value is called as Global Minima.
'''

import pandas as pd

# LINEAR REGRESSION MODEL

# EXAMPLE: MLR - Multy Linear Regression

# 1. Reading Data

data = pd.read_csv('Ecommerce Customers.csv')
# print(data

df = pd.DataFrame(data)
# print(df)

drop = df.drop(['Email','Address','Avatar'], axis=1)

# 2. Separating Data in Dependent and Independent Variable.

X = df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]             # Independent Feature

y = df['Yearly Amount Spent']               # Dependent Feature

# 3. Divide Datasets X and y for Training set and Testing Set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

# 4. Build a Linear Regression Model

from sklearn.linear_model import LinearRegression

Linear_model = LinearRegression()

Model = Linear_model.fit(x_train, y_train)          # training to model
print(Model)

# 5. Predict Output

y_predict = Model.predict(x_test)
print(y_predict)

# 6. Accuracy of Model in Percentage

Accuracy_in_percentage = Model.score(x_train, y_train)
print(Accuracy_in_percentage)

# 7. Check Coefficient of Intercept(c) and Coefficient of Slope(m)

Slope_m = Model.coef_
print(Slope_m)

Constant_c = Model.intercept_
print(Constant_c)

# 8. Final prediction Manually vs Model

Final_Manually = (26.09396884*33.99257277495374) + (38.96192895*13.338975447662111)+(0.1154769*-37.22580613162114)+(62.22698227*2.482607770510596) + -1056.7659583518264
# print(Final_Manually)

Final_Model = Model.predict([[33.87103787934198,12.026925339755058,34.47687762925054,5.493507201364199]])
print(Final_Model)

# 9. Find error in prediction using R-Squared

from sklearn.metrics import r2_score

R_Squ = r2_score(y_test, y_predict)
print(R_Squ)

# 10. Find error in prediction using Adjusted R-Squared over R-Squared

# Terminal -> pip install statsmodels -> enter

import statsmodels.api as sm

X = sm.add_constant(X)
final = sm.OLS(y, X).fit()
Adjusted_R_Square = final.rsquared_adj
print(Adjusted_R_Square)

'''☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻'''

# COEFFICIENT OF DETERMINATION / R-SQUARED
'''
- Once we build a model we have to see how good the model is or how reliable the model is, for that we use another matric
  called as Coefficient of Determination and this is represented as R-Squared.
- R-Squared ranges between 0 & 1.
- R-Squared is the ratio between variance explained by your model to the total variance in the datapoints.
- R-Squared values is a statistical measure of how close the datapoints are to the fitted line.
- High R-Squared value is considered as good linear model.
- R-Squared = 1 - [ SSres / SStot ] = 1 - [ sum of (Yi - Yi.cap)square / sum of (Yi - Y.bar)square ]
'''

# Problems with R-squared statistic
'''
- The R-squared statistic isn’t perfect. In fact, it suffers from a major flaw. Its value never decreases no matter the 
  number of variables we add to our regression model. That is, even if we are adding redundant variables to the data, 
  the value of R-squared does not decrease. It either remains the same or increases with the addition of new independent 
  variables. This clearly does not make sense because some of the independent variables might not be useful in determining 
  the target variable. 
- Adjusted R-squared deals with this issue.
'''

# ADJUSTED R-SQUARE
'''
- Compared to R-Squared which can only increase, Adjusted R-Squared has the capability to decrease with the addition of 
  less significant variables, thus resulting in a more reliable and accurate evaluation.
- Adjusted R-Square = 1 - [ ( 1 - R.square) ( n-1 ) / ( n-k-1) ]
    n = total sample size or no. of datapoints in our dataset
    k = no. of independent variables
    R = R-Square value
- NOTE: ADJUSTED R-SQUARED VALUE IS ALWAYS LESS THAN THAT OF THE R-SQUARED
'''

'''☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻'''
