import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# HANDLING IMBALANCE DATASET
'''
- Imbalanced data refers to those types of datasets where the target class has an uneven distribution of observations, 
  i.e one class label has a very high number of observations (majority class) and the other has a very low number of 
  observations (minority class).
- To balance the data we use Resampling Techinques.
- Resampling technique is used to upsample and downsample the minority and majority class.
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 1. OVERSAMPLING
'''
- Oversampling increases the number of observations to the minority class by randomly replicating them in order to meet 
  as same as the majority class.
'''

# EXAMPLE:
'''
- Total obeservations = 1000 
- Non-Cancer Patients = 910
- Cancer Patients = 90
- In Oversampling we replicating Cancer Patients by 10 times to meet the majority class.
- 90 * 10 = 900, now we have total observations are 1810
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 2. UNDERSAMPLING
'''
- Undersampling decreases the number of observations to the majority class by randomly deleting them in order to meet 
  as same as the minority class.
'''

# EXAMPLE:
'''
- Total obeservations = 1000 
- Non-Cancer Patients = 910
- Cancer Patients = 90
- In Undersampling we decreasing the Non-Cancer Patients by 0.10 times of it to meet the minority class.
- 910 * 0.10 = 91, now we have total observations are 181
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# pip install imblearn

# CHECKING DATASET FOR RESAMPLING
'''
data = pd.read_csv('aug_train.csv')
df = pd.DataFrame(data)

Test = df['target']
print(Test)

plot = sns.countplot(df['target'])
plt.show()

value_0 = df[df['target'] == 0]
value_1 = df[df['target'] == 1]

print(value_0.shape)
print(value_1.shape)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# EXAMPLE: 1 - OVERSAMPLING

data = pd.read_csv('aug_train.csv')
df = pd.DataFrame(data)
print(df.shape)
X = df.drop('target', axis=1)
y = df['target']

from imblearn.over_sampling import RandomOverSampler

Over = RandomOverSampler(sampling_strategy=0.5)

X_over, y_over = Over.fit_resample(X, y)

print(X_over.shape)
print(y_over.shape)
''''''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# EXAMPLE: 2 - UNDERSAMPLING

data = pd.read_csv('aug_train.csv')
df = pd.DataFrame(data)
print(df.shape)

X = df.drop('target', axis=1)
y = df['target']

from imblearn.under_sampling import RandomUnderSampler

Under = RandomUnderSampler(sampling_strategy=0.5)

X_under, y_under = Under.fit_resample(X, y)

print(X_under.shape)
print(y_under.shape)
''''''
# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻