
# WHAT IS LOGISTIC REGRESSION ALGORITHM ?
'''
- Logistic Regression is one of the most popular Machine Learning algorithm which comes under the Supervised Learning
  technique. It is used for predicting the Categorical dependent variable using a given set of independent variables.
- Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. 
  but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.
- The value of the logistic regression must be between 0 and 1, which cannot go beyond this limit, so it forms a curve 
  like the "S" form. The S-form curve is called the SIGMOID FUNCTION or the LOGISTIC FUNCTION.
- In logistic regression, we use the concept of the threshold value, which defines the probability of either 0 or 1. 
  Such as values above the threshold value tends to 1, and a value below the threshold values tends to 0.
'''

import pandas as pd

# LOGISTIC REGRESSION MODEL

# EXAMPLE:

# 1. Reading Data

data = pd.read_csv('Heart.csv')
# print(data)

df = pd.DataFrame(data)
# print(df)

# 2. Separating Data in Dependent and Independent Variable.

X = df.drop(['target'], axis=1)

y = df.target

# 3. Divide Datasets X and y for Training set and Testing Set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 4. Build a Logistic Regression Model

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

model = LR.fit(x_train, y_train)

# 5. Predict Output

y_predict = model.predict(x_test)

# 6. Evaluation of Model

# 6.1 Accuracy of Model in Percentage

accuracy = model.score(x_test, y_test)
print(accuracy)

# 6.2 Confusion Matrix

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,classification_report

matrix = confusion_matrix(y_test, y_predict)
print(matrix)

# 6.3 Precision Score

Precision_Score = precision_score(y_test, y_predict)
print(Precision_Score)

# 6.4 Recall Score

Recall_Score = recall_score(y_test, y_predict)
print(Recall_Score)

# 6.5 F1 Score

F1_Score = f1_score(y_test, y_predict)
print(F1_Score)

# 7. Classification Report

Classification_Report = classification_report(y_test, y_predict, labels=[1,0])
print(Classification_Report)

'''☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻'''

# CONFUSION MATRIX
'''
- A Confusion matrix is an N x N matrix used for evaluating the performance of a classification model, where N is the 
  number of target classes. The matrix compares the actual target values with those predicted by the machine learning model.
- For the 2 prediction classes of classifiers, the matrix is of 2*2 table, for 3 classes, it is 3*3 table, and so on.
- The matrix is divided into two dimensions, that are PREDICTED VALUES and ACTUAL VALUES along with the total number of 
  predictions.
- If we have an imbalanced dataset to work with, it’s always better to use confusion matrix.
- A good model is one which has high TP and TN rates, while low FP and FN rates. Let see about it,

- TRUE POSITIVE (TP) 
    • The predicted value matches the actual value
    • The actual value was positive and the model predicted a positive value
- TRUE NEGATIVE (TN) 
    • The predicted value matches the actual value
    • The actual value was negative and the model predicted a negative value
- FALSE POSITIVE (FP) – Type 1 error
    • The predicted value was falsely predicted
    • The actual value was negative but the model predicted a positive value
    • Also known as the Type 1 error
- FALSE NEGATIVE (FN) – Type 2 error
    • The predicted value was falsely predicted
    • The actual value was positive but the model predicted a negative value
    • Also known as the Type 2 error

- Actual Values: True
                 False
- Predicted Values: Positive
                    Negative
- ACCURACY = (TP + TN) / (TP + TN + FP + FN)
'''

# PRECISION
'''
Out of all positive classes that have predicted correctly by the model, how many of them were actually true.

PRECISION = TP / (TP + FP)
'''

# RECALL
'''
It is defined as the out of total positive classes, how our model predicted correctly. 
The recall must be as high as possible.

RECALL = TP / (TP + FN)
'''

# F1-SCORE
'''
- It is difficult to compare two models with low precision and high recall or vice versa. So to make them comparable, 
  we use F1-Score. F1-score helps to measure Recall and Precision at the same time.
- The F1 score is a number between 0 and 1 and is the harmonic mean of precision and recall.

F1-SCORE = 2PR / (P + R)
'''

# WHEN TO USE ACCURACY / PRECISION / RECALL / FI-SCORE
'''
a. Accuracy is used when the True Positives and True Negatives are more important. Accuracy is a better metric for 
   Balanced Data.

b. Whenever False Positive is much more important, use Precision.

c. Whenever False Negative is much more important, use Recall.

d. F1-Score is used when the False Negatives and False Positives are important. F1-Score is a better metric for 
   Imbalanced Data.
'''




'''☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻'''