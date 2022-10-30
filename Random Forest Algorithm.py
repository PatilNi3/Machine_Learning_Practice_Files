
# WHAT IS RANDOM FOREST ALGORITHM ?
'''
- Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be
  used for both Classification and Regression problems in ML. It is based on the concept of ensemble learning, which is
  a process of combining multiple classifiers to solve a complex problem and to improve the performance of the model.
- Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and
  takes the average to improve the predictive accuracy of that dataset.
- The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
- It builds decision trees on different samples and takes their majority vote for classification and average in case of
  regression.
'''

# TYPES OF ENSEMBLE TECHNIQUE:
'''
1. Bagging– It creates a different training subset from sample training data with replacement & the final output is 
   based on majority voting. For example,  Random Forest.

2. Boosting– It combines weak learners into strong learners by creating sequential models such that the final model has 
   the highest accuracy. For example,  ADA BOOST, XG BOOST.
   
- Random Forest works on Bagging principle.
- Bagging = B + agg = Bootstrap + Aggrigation
'''

# APPLICATIONS OF RANDOM FOREST:
'''
1. Banking
2. Health Care Services
3. Stock Exchange 
4. E-Commerce
'''

# RANDOM FOREST ALGORITHM MODEL

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read Dataset

df = pd.read_csv('heart.csv')

# 2. Plot Dependent Variable

sns.countplot(df['target'])
# plt.show()

# 3. Split dataset into X and y

X = df.drop('target', axis=1)

y = df['target']

# 4. Split dataset for Training and Testing

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 5. Build Model

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators=150)                # n_estimators– number of trees the algorithm builds before averaging the predictions.

model = RFC.fit(x_train, y_train)

# 6. Predict Target w.r.t. x_test

y_pred = model.predict(x_test)

# 7. Accuracy of Model in Percentage

score = model.score(x_test, y_test)
print(score)

# 8. Model Tuning - Hyper Parameter Optimization

from sklearn.model_selection import GridSearchCV

Parameter_RFC = {'n_estimators':[50, 100, 150, 200]}

GS = GridSearchCV(RFC, Parameter_RFC, cv=5)         # cv = cross validation = dataset spliting

model1 = GS.fit(x_train, y_train)

best = GS.best_params_         # best parameter out of [50,100,150,200]
print(best)

# 9. Validation of Model

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

score1 = accuracy_score(y_pred, y_test)
print(score1)

# 10. Prediction of model

pred = RFC.predict([[58,1,0,125,300,0,0,171,0,0,2,2,3]])
print(pred)

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻