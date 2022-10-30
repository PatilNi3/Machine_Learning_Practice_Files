
# WHAT IS DECISION TREE ALGORITHM ?
'''
- Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but
  mostly it is preferred for solving Classification problems.
- It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the
  decision rules and each leaf node represents the outcome.
- In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make
  any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any
  further branches.
- In order to build a tree, we use the CART Algorithm, which stands for Classification and Regression Tree algorithm.
• Root Node: Root node is from where the decision tree starts. It represents the entire dataset, which further gets
  divided into two or more homogeneous sets.
• Leaf Node: Leaf nodes are the final output node, and the tree cannot be segregated further after getting a leaf node.
'''

import pandas as pd

# MANUAL CALCULATIONS:
'''
data = pd.read_csv('Decision Tree.csv')
df = pd.DataFrame(data)
print(df)

print("-----------------------------------------------------")
Sunny = df.loc[[0, 1, 7, 8, 10]]
print(Sunny)
print("-----------------------------------------------------")
Overcast = df.loc[[2, 6, 11, 12]]
print(Overcast)
print("-----------------------------------------------------")
Rain = df.loc[[3, 4, 5, 9, 13]]
print(Rain)
print("-----------------------------------------------------")
'''

# ACTUAL PROGRAMMING:

# 1. Read Dataset

data = pd.read_csv('Decision Tree.csv')

df = pd.DataFrame(data)
# print(df)

# 2. Encode Features

Outlook_Encode = df['Outlook'].map({'Sunny':1, 'Overcast':2, 'Rain':3})
# print(Outlook_Encode)

Temp_Encode = df['Temp'].map({'Hot':1, 'Mild':2, 'Cool':3})
# print(Temp_Encode)

Humidity_Encode = df['Humidity'].map({'High':1, 'Normal': 2})
# print(Humidity_Encode)

Wind_Encode = df['Wind'].map({'Weak':1, 'Strong':2})
# print(Wind_Encode)

PlayTennis_Encode = df['Play Tennis'].map({'Yes':1, 'No':0})
# print(PlayTennis_Encode)

Updated_df = pd.concat([Outlook_Encode, Temp_Encode, Humidity_Encode, Wind_Encode, PlayTennis_Encode], axis=1)
print(Updated_df)

# 3. Split dataset into X and y

X  = Updated_df.drop('Play Tennis', axis=1)

y = Updated_df['Play Tennis']

# 4. Split dataset for Training and Testing

from  sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 5. Build Model

# 5.1. Build model on Information Gain

from sklearn.tree import DecisionTreeClassifier

Entropy_Model = DecisionTreeClassifier(criterion='entropy')

model1 = Entropy_Model.fit(x_train, y_train)

# 5.2. Build model on Gini Index

from sklearn.tree import DecisionTreeClassifier

Gini_Model = DecisionTreeClassifier(criterion='gini')

model2 = Gini_Model.fit(x_train, y_train)

# 6. Predict Play Tennis w.r.t. x_test for both models

y_pred1 = model1.predict(x_test)

y_pred2 = model2.predict(x_test)

# 7. Validation of Model

# 7.1. For Information Gain model

from sklearn.metrics import confusion_matrix, accuracy_score

matrix1 = confusion_matrix(y_test, y_pred1)
print(matrix1)

accuracy1 = accuracy_score(y_test, y_pred1)
print(accuracy1)

# 7.2. For Gini Index model

from sklearn.metrics import confusion_matrix, accuracy_score

matrix2 = confusion_matrix(y_test, y_pred1)
print(matrix2)

accuracy2 = accuracy_score(y_test, y_pred1)
print(accuracy2)

# 8. Plot Decision Tree model

# 8.1. For Information Gain

from sklearn.tree import plot_tree

plot_model = plot_tree(decision_tree=model1)
print(plot_model)

import  matplotlib.pyplot as plt

plt.show()

# 8.2. For Gini Index

from sklearn.tree import plot_tree

plot_model = plot_tree(decision_tree=model2)
print(plot_model)

import  matplotlib.pyplot as plt

plt.show()

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# ENTROPY:
'''
- Entropy is the measure of uncertainty. It is a metric that measures the impurity of something.
- Entropy(s)=-P(yes)logP(yes)-P(no)logP(no)

where,

s = total sample space

P(yes) = Probability of yes
P(no) = Probability of no

If number of YES = number of NO, then

    P(s)=0.5 and Entropy(s) = 1

If it contains either all YES or all NO, Then

    P(s) = 1 or 0 and Entropy(s) = 0
'''

# INFORMATION GAIN:
'''
- Information gain is the measure of how much information a feature gives about the class. It is the decrease in entropy 
  after splitting the dataset based on the attribute. 
- Constructing a decision tree is all about finding the attribute that has the highest information gain.
- Information Gain= Entropy(S)- [(Weighted Avg) * Entropy(each feature)
'''

# GINI INDEX:
'''
- Gini Index is a measure of impurity or purity used while creating decision tree in the CART algorithm.
- An attribute with the low gini index  should be preferred as compared to the high gini index.
- It only creates binary splits and the CART algorithm uses the gini index to create binary splits.
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻