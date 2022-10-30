### 06/06/2022 ### 08:00PM to 09:30PM ###

# WHAT IS MACHINE LEARNING ?
'''
- Machine Learning enables a machine to automatically learn from data, improve performance from experiences and predict things without being explicitly programmed.
- Building mathematical models and making predictions using historical data or information.
- A Machine Learning system learns from hostorical data, builds the prediction models and whenever it receives new data, predicts the output for it.
'''

# WHAT IS ARTIFICIAL INTELLIGENCE ?
'''
- Artificial intelligence is a technique which allows machines to mimic human behaviour.
- AI refers to a broader idea where machines can execute tasks "smartly."
'''

# WHAT IS DEEP LEARNING ?
'''
- Deep learning is a particular kind of machine learning that is inspired by the functionality of our brain sells called 
  Neurons which led to the concept of Artificial Neural Network.
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# Real time Examples of Machine Learning
'''
1. Image Recognition (Face recognition)
2. Speeech Recognition (Amazon Alexa, Google Home)
3. Medical Recognition (Rare diseases, Genetic diseases)
4. Statistical Arbitrage (strategy used in finance)
5. Predictive Analysis (product development to real estate pricing)
6. Extraction (extract structured data from unstructured)
'''

# Data from different different sources
'''
1. API
2. AWS Cloud
3. Test Environment
4. Local Database
5. CSV (99% used because of light weight)
6. Sharepoint
'''

# For Data Analysis we using different different libraries
'''
1. Pandas
2. NumPy
3. SeaBorn
4. MatPlotLib
5. SciPy
6. ScikitLearn
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# CLASSIFICATION OF MACHINE LEARNING
'''
1. Supervised
2. Un-supervised 
3. Reinforcement
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 1. SUPERVISED LEARNING
'''
- Supervised learning is a type of machine learning method in which we provide sample labeled data to the machine 
  learning system in order to train it, and on that basis, it predicts the output.
- The system creates a model using labeled data to understand the datasets and learn about each data, once the training 
  and processing are done then we test the model by providing a sample data to check whether it is predicting the exact 
  output or not.
- The goal of supervised learning is to map input data with the output data. The supervised learning is based on 
  supervision and it is the same as when a student learns things in the supervision of the teacher. The example of 
  supervised learning is spam filtering.
- Y = f(X) -> Y = Output(prediction), X = Input(test)
'''

# Supervised learning can be grouped further in two categories of algorithms:
'''
1. Regression - If target feature are always in coninues form.
2. Classification - If target feature are always in categorical form.
'''

# 1.1: Regression Algorithms
'''
- Regression analysis helps in the prediction of a continuous variable.
- Linear regression shows the linear relationship between the independent variable (X-axis) and the dependent 
  variable (Y-axis), hence called linear regression.
- Regression Algorithms can be further divided into the Mainly two category:

  1. If there is only one independent input variable (X), then such linear regression is called SIMPLE LINEAR REGRESSION.
  2. If there is more than one independent input variable (X1, X2, ... ), then such linear regression is called MULTIPLE L
  INEAR REGRESSION.
'''

# 1.2: Classification Algorithms
'''
- If we have to predict the categorical values, we need Classification algorithms. Such as, YES or NO, 0 or 1, Spam or 
  Not Spam, cat or dog, etc.
- The best example of an ML classification algorithm is Email Spam Detector.

- Classification Algorithms can be further divided into the Mainly 7 category

  1. Logistic Regression - Logistic regression algorithm works with the categorical variable such as 0 or 1, Yes or No, 
     True or False, Spam or not spam, etc.
  2. Support Vectore Machine - Support Vector Machine is a supervised learning algorithm which can be used for regression 
     as well as classification problems. So if we use it for regression problems, then it is termed as Support Vector 
     Regression. Support Vector Regression is a regression algorithm which works for continuous variables
  3. K-Nearest Neighbours - K-nearest neighbors is one of the most basic yet important classification algorithms in 
     machine learning and have several applications in pattern recognition, data mining, and intrusion detection.
  4. Kernel SVM - Kernel Function is a method used to take data as input and transform it into the required form of 
     processing data. “Kernel” is used due to a set of mathematical functions used in Support Vector Machine
  5. Naive Bayes - Naive Bayes is one of the powerful machine learning algorithms that is used for classification. 
     It is an extension of the Bayes theorem wherein each feature assumes independence. It is used for a variety of tasks 
     such as spam filtering and other areas of text classification.
  6. Decision Tree Classification - Decision Tree is a supervised learning algorithm which can be used for solving both 
     classification and regression problems. It can solve problems for both categorical and numerical data
  7. Random Forest Classification - It is based on the concept of ensemble learning, which is a process of combining 
     multiple classifiers to solve a complex problem and to improve the performance of the model.
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 2. UNSUPERVISED LEARNING
'''
- Un-supervised learning is a methods in which Machine learns without an supervision
- The training is provided to the machine with the set of data that has not been labeled, classified, or categorized, 
  and the algorithm needs to act on that data without any supervision. The goal of unsupervised learning is to 
  restructure the input data into new features or a group of objects with similar patterns.
- In unsupervised learning, we don't have target feature.
'''

# Un-supervised learning can be grouped further in two categories of algorithms:
'''
1. Clustering
2. Association
'''

# 2.1: Clustering
'''
- Clustering is a task of dividing the data sets into a certain number of clusters in such a manner that the data points 
  belonging to a cluster have similar characteristics. 
- Clusters are nothing but the grouping of data points such that the distance between the data points within the clusters 
  is minimal.
  
  1. K-means clustering - K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. 
     Here K defines the number of pre-defined clusters that need to be created in the process, as if K=2, there will be 
     two clusters, and for K=3, there will be three clusters, and so on. 
     The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the 
     process until it does not find the best clusters. The value of k should be predetermined in this algorithm.
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 3. REINFORCEMENT LEARNING
'''
- Reinforcement learning is a feedback-based learning method, in which a learning agent gets a reward for each right 
  action and gets a penalty for each wrong action. The agent learns automatically with these feedbacks and improves its 
  performance.
- The goal of an agent is to get the most reward points and hence it improves its performance.
- Example: Robotics (Robo Dog), AI (Chess Game)
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# MATHEMATICS FOR MACHINE LEARNING ALGORITHMS AND DATA SCIENCE
'''
- Mathematics is the core of designing ML algorithms that can automatically learn from data and make predictions. 
  Therefere, it is very important to understand the Maths before going into the deep understanding of ML algorithms.
- Learning mathematics in machine learning is not about solving a maths problem, rather understanding the application of 
  maths in ML algorithms and their working.
'''

# 1. MEDIAN
'''
- Median is nothing but middle value from ourr dataset when datapoints are arranged with ascending order.
- If we have outliers in our dataset it is best choice to use Median because mean can be affected by outliers.
- Outliers - Outliers is nothing but datapoints which are very different from rest of the datapoints.
- Median allows you to come with right guess while filling null values.
- When number of datapoints are even, then take mean of middle two values.
- Use cases of Median-
    1) In simple Descriptive Statistics.
    2) Handling Missing values from dataset.
'''

# 2. MEAN
'''
- Mean is nothing but the average value of the given numbers or data.
- If our data is balanced and does not contains any outliers then we can use mean for filling missing values from dataset.
- To calculate the mean, we need to add the total values given in a datasheet and then divide the sum by the total number 
  of values.
- Use cases of Mean-
    1) In simple Descriptive Statistics.
    2) Handling Missing values from dataset.
'''

# 3. MODE
'''
- A mode is defined as the value that has a higher frequency in a given set of values. It is the value that appears the 
  most number of times.
  Example: In the given set of data: 2, 4, 5, 5, 6, 7 the mode of the dataset is 5 since it has appeared in the set twice.
- Use cases of Mode-
    1) In simple Descriptive Statistics.
    2) Fill issing values from categorical features.
'''

# 4. PERCENTILE
'''
- Percentile are used to understand and interpret data.
- In everyday life, percentiles are used to understand values such as test score, health indicator, battery indicator 
  and other measurments.
- For example: 
    • 50th percentile >>> It means 50% datapoints are at left side and right side of this point.
    • 25th percentile >>> It means 25% datapoints are at left side and 75% datapoints are at right side of this point.
    • 75th percentile >>> It means 75% datapoints are at left side and 25% datapoints are at right side of this point.
    • The range between 25th to 75th percentile is called as Interquartile Range.
'''

# 5. VARIANCE
'''
- The variance measures how far the data points are spread out from the average value and is equal to the sum of squares 
  of diffences between the data values and the average (the mean).
'''

# 6. STANDARD DEVIATION
'''
- The standard deviation is simply the square root of the variance and measures the extent to which data varies from its 
  mean.
- Standard deviation is often preferred over the variance because it has the same unit as the data points, which means 
  you can interpret it more easily.
- The standard deviation defined by sigma.
'''


# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻