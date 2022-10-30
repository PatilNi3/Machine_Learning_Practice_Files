
# WHA IS PANDAS ?
'''
- Pandas is defined as an open-source library that provides high-performance data manipulation in python. The name of
  Pandas is derived from the word Panel Data, which means an Econometrics from Multidimensional Data.
- Data analysis requires lots of processing, such as restructuring, cleaning or merging etc. There are different tools
  available for fast data processing, such as NumPy, SciPy, Cython and Pandas. But we prefer Pandas because working with
  Pandas is fast, simple and more expressive than other tools.
- Pandas is to perform or to construct data in tabular format is called Pandas Framework
- pip install pandas
'''

import pandas as pd

# Different types of Data Structures in Pandas
'''
1. 1D - Series
2. 2D - DataFrames
3. 3D - Panel
'''

# WHAT IS DATAFRAME ?
'''
- A Series is essentially a column, and a DataFrame is a multi-dimensional table made up of a collection of Series.
- A Pandas DataFrame is a 2 dimensional data structure, like a 2 dimensional array, or a table with rows and columns.
  Pandas DataFrame consist of three principal components - data, rows and columns
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# READ DATA IN THE FORM OF DATAFRAME
'''
- Pandas DataFrame is a widely used data structure which works with a two-dimensional arraywith labeled axes (rows and 
  columns). DataFrame is defined as a standard way to store data that has two different indexes, i.e. row index and column
  index. It consist of following properties:
  • The columns can be heterogeneous types like int, bool and so on.
  • It can be seen as a dictionary of Series structure where both the rows and columns are indexed. It is denoted as 
    "columns" in case of columns and "index" in case of rows.
'''

# 1. READ DATA BY USING CSV
'''
DF = pd.read_csv('Automobile_data.csv')
print(DF)
print(type(DF))      # <class 'pandas.core.frame.DataFrame'>
'''

# 2. READ DATA BY USING EXCEL

# EXAMPLE: 1
'''
DF1 = pd.read_excel('AboutUS.xlsx')
print(DF1)
print(type(DF1))
'''

# EXAMPLE: 2
'''
DF1 = pd.read_excel('AboutUS.xlsx', sheet_name='Auto1')
print(DF1)
print(type(DF1))
'''

# 3. READ DATA FROM DICTIONARY

# EXAMPLE: 1
'''
DICTIONARY = {'Id':[1,2,3,4], 'Name':['Nitin', 'Neha', 'Ruchita', 'Timish'], 'City':['Banglore', 'Sydney', 'Mumbai', 'Mumbai']}
DF2 = pd.DataFrame(DICTIONARY)
print(DF2)
'''

# EXAMPLE: 2 -LIST OF DICTIONARY
'''
DICTIONARY = [{'Name':'Nitin', 'Designation':'ML Engineer', 'Salary':150000},
              {'Name':'Neha', 'Designation':'Data Analyst', 'Salary':21000},
              {'Name':'Ruchita', 'Designation':'Data Enginner', 'Salary':60000},
              {'Name':'Timish', 'Designation':'Data Scientist', 'Salary': 100000}]

DF3 = pd.DataFrame(DICTIONARY)
print(DF3)
'''

# 4. CREATING DATAFRAME BY USING TUPLE LIST
'''
TUPLE_LIST = [('Nitin', 26, 'Banglore', 'nitinpatilp29@gmail.com'),
              ('Neha', 27, 'Sydney', 'nehabendalep29@gmail.com'),
              ('Ruchita', 30, 'Mumbai', 'ruchitapatil.p29@gmail.com'),
              ('Timish', 25, 'Mumbai', 'timishbhau@gmail.com')]

DF4 = pd.DataFrame(TUPLE_LIST)
print(DF4)

DF4 = pd.DataFrame(data=TUPLE_LIST)
print(DF4)

DF4 = pd.DataFrame(data=TUPLE_LIST, columns=['NAME', 'AGE', 'CITY', 'EMAIL'])
print(DF4)

DF4 = pd.DataFrame(data=TUPLE_LIST, index=['#1', '#2', '#3', '#4'], columns=['NAME', 'AGE', 'CITY', 'EMAIL'])
print(DF4)

DF4.to_json('TUPLE_LIST.json')
'''

# 5. CREATING DATAFRAME BY USING TEXT FILE
'''
DF5 = pd.read_csv('Automobile_data.txt')
print(DF5)
print(type(DF5))
df = pd.DataFrame(DF5)
print(df)
print(type(df))
'''

# 6. READ DATA FROM JSON
'''
DF6 = pd.read_json('TUPLE_LIST.json')
print(DF6)
'''

# 7. READ DATA USING WEB SCRAPING
'''
DF7 = pd.read_html('https://en.wikipedia.org/wiki/Automotive_industry')
print(DF7[1])
print(type(DF7))
'''

# Similarly, we can read data from API, DataBase, S3 Bucket

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# SAVE THE DATAFRAMES IN DIFFERENT DIFFERENT FORMAT

# To CSV
'''
DF4.to_csv('AboutUS.csv')

DF4.to_csv('AboutUS1.csv', index=False)
'''

# To EXCEL
'''
DF4.to_excel("AboutUS.xlsx")

DF4.to_excel("AboutUS1.xlsx", index=False)
'''

# To DICTIONARY
'''
DICT = DF4.to_dict()
print(DICT)
'''

# To HTML
'''
HTML = DF4.to_html()
print(HTML)
'''

# To JSON
'''
JSON = DF4.to_json()
print(JSON)
'''

# To PICKLE
'''
PICKLE = DF4.to_pickle('PICKLE.pickle')
print(PICKLE)
'''

# To SQL
'''
SQL = DF4.to_sql
print(SQL)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes # # DTypes #
# '''This returns a Series with the data type of each column.'''
#
# # N16 = N.dtypes
# # print(N16)
#
# # N17 = N4.dtypes
# # print(N17)

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# MOST IMPORTANT OPERATIONS ON DATAFRAME

# 1. FETCH COLUMNS AND ROWS FROM DATAFRAME
'''
print(DF.shape)
'''

# 2. FETCH SPECIFIC COLUMN FROM DATAFRAME
'''
print(DF['make'])
'''

# 3. FETCH MULTIPLE COLUMNS FROM DATAFRAME
'''
print(DF[['make', 'horsepower', 'price']])
'''

# 4. LIST OF COLUMNS FROM DATAFRAME
'''
print(DF.columns)
'''

# 5. ADD NEW COLUMN IN DATAFRAME

# EXAMPLE: 1
'''
NC = DF['owner']='self'
print(NC)
print(DF)
'''

# EXAMPLE: 2
'''
import datetime
DF['Purchase_Date'] = datetime.datetime.now()
print(DF)
'''

# 6. CREATE NEW DATAFRAME FROM USING SELECTED COLUMNS OF EXISTING DATAFRAME
'''
DF8 = DF[['make', 'horsepower', 'price']]
print(DF8)
'''

# 7. CLEAN DATA USING LAMBDA FUNCTION
'''
DF9 = DF['New_make'] = DF['make'].apply(lambda x:x.split('-')[0])
print(DF9)
'''

# 8. OPERATIONS ON SPECIFIC COLUMN
'''
DF10 = DF['New_price'] = DF['price'].apply(lambda x:x*1.1)
print(DF10)
'''

# 9. ENCODING COLUMNS

    # 9.1. Using Lambda Function
'''
DF11 = DF['Encoded_fuel-type'] = DF['fuel-type'].apply(lambda x:1 if x=='gas' else 0)
print(DF11)
'''

    # 9.2. Using Map Function
'''
DF11 = DF['Encoded_fuel-type'] = DF['fuel-type'].map({'gas':1, 'diesel':0})
print(DF11)
'''

    # 9.3. Using Normal Function
'''
def encode(x):
    if x=='gas':
        return 1
    else:
        return 0

DF11 = DF['Encoded_fuel-type'] = DF['fuel-type'].apply(encode)
print(DF11)
'''

# 10. DELETE COLUMN FROM DATAFRAME
'''
DF12 = DF.drop('symboling', axis=1)
print(DF12)

- 1) 0 = Interpreter will search given column name in rows of DataFrame.
- 2) 1 = Interpreter will search given column name in column of DataFrame.

- 1) inplace = True = Will commit changes on DataFrame.
- 2) inplace = False = Changes will appear but wont commit.
'''

# 11. DELETE MULTIPLE COLUMNS FROM DATAFRAME
'''
DF13 = DF.drop(columns=['symboling', 'normalized-losses', 'drive-wheels'], axis=1)
print(DF13)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# DATA DEFINITION OPERATIONS
''''''
DATA = pd.read_csv('Null_Dataset.csv.txt')
# print(DATA)
DF1 = pd.DataFrame(DATA)
# print(DF1)

# 1. FIND NULL VALUES FROM COLUMNS

# EXAMPLE: 1
'''
print(DF1.isnull())
'''

# EXAMPLE: 2
'''
print(DF1.isnull().sum())
'''

# 2. TO FIND UNIQUE VALUES FROM SPECIFIC COLUMNS
'''
print(DF1['Name'].unique())

print(DF1['City'].unique())
'''

# 3. TO FIND DATA TYPE OF EACH COLUMN
'''
print(DF1.dtypes)
'''

# 4. CHANGE DATA TYPE OF EXISTING COLUMN
'''
DF1['Age'] = DF1['Age'].astype('object')
print(DF1.dtypes)
'''

# 5. DELETE ROWS WHICH HAVE NULL VALUES
'''
Pandas = DF1.dropna()
print(Pandas.shape)
'''

# 6. DROP NULL VALUES FROM ROWS or COLUMNS WITH CONDITIONS
'''
Pandas = DF1.dropna(axis=0, how='any', subset=['Salary'])
print(Pandas)
print(Pandas.shape)
'''

# 7. REPLACE NONE VALUES FROM DATAFRAME

'''
import numpy as np
Pandas = DF1.replace(np.NaN, 'ML')
print(Pandas)
'''

# 8. WE CAN USE REPLACE ENCODING OF COLUMN
'''
Pandas = DF1.replace(['BANGLORE', 'PUNE', 'MUMBAI', 'HYDERABAD', 'GURGAON', 'CHENNAI', 'KOLKATA'],[1, 2, 3, 4, 5, 6, 7])
print(Pandas)
'''

# 9. USE GROUP BY CLAUSE
'''
Pandas = DF1.groupby('City')
print(Pandas)                   # GIVES OBJECT ID
print(Pandas.size())
'''

# 10. RENAMING EXISTING COLUMN
'''
Pandas = DF1.rename({'Name':'New_Name', 'Age':'New_Age', 'City':"New_City", 'Salary':'New_Salary'}, axis=1)
print(Pandas)
'''

# 11. CONCATINATION OF DATAFRAME

DF2 = pd.read_csv('scaling.csv')
DF3 = pd.read_csv('table.csv')

# EXAMPLE: 1
'''
DF4 = pd.concat([DF2, DF3])
print(DF4)
'''

# EXAMPLE: 2
'''
DF5 = pd.concat([DF2, DF3], axis=1)
print(DF5)
'''

# EXAMPLE: 3
'''
Favourite1 = {'Name':['Nitin', 'Neha', 'Ruchi'],
             'Watch':['Jaeger LeCoultre', 'Daniel Wellington', 'IWC Schaffhausen'],
             'Shoes': ['Nike Jordan', 'Jimmy Choo', 'Dolce & Gabbana'],
             'Gadgets':['I-Pad', 'Macbook', 'DJI Drone']}

df1 = pd.DataFrame(Favourite1)

Favourite2 = {'Name':['Nitin', 'Neha', 'Ruchi'],
             'Watch':['Titan', 'Fossil', 'Fastrack'],
             'Shoes': ['Puma', 'H & M', 'Red Tape'],
             'Gadgets':['I-Phone', 'Gaming PC', 'Play Station']}

df2 = pd.DataFrame(Favourite2)


Stuff = pd.concat([df1, df2])
print(Stuff)

Stuff1 = pd.concat([df1, df2], ignore_index=True)
print(Stuff1)

Stuff2 = pd.concat([df1, df2], ignore_index=False, keys=['Your_Favourite', 'My_Favourite'])
print(Stuff2)

Stuff3 = pd.concat([df1, df2], axis=1)                              # new Column added
print(Stuff3)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# AGGREGATE FUNCTIONS

# 1. MAX FUNCTION

# EXAMPLE: 1
'''
Pandas = DF3.max()
print(Pandas)
'''

# EXAMPLE: 2
'''
Pandas = DF3['age-months'].max()
print(Pandas)
'''

# 2. MIN FUNCTION

# EXAMPLE: 1
'''
Pandas = DF3.min()
print(Pandas)
'''

# EXAMPLE: 2
'''
Pandas = DF3['age-months'].min()
print(Pandas)
'''

# 3. MEAN FUNCTION

# EXAMPLE: 1
'''
Pandas = DF3.mean()
print(Pandas)
'''

# EXAMPLE: 2
'''
Pandas = DF3['age-months'].mean()
print(Pandas)
'''

# 4. STANDARD DEVIATION OF COLUMNS

# EXAMPLE: 1
'''
Pandas = DF3.std()
print(Pandas)
'''

# EXAMPLE: 2
'''
Pandas = DF3['age-months'].std()
print(Pandas)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# COMPARISON OPERATOR

# EXAMPLE: 1
'''
Pandas = DF3[DF3['age-months']==85]
print(Pandas)
'''

# EXAMPLE: 2
'''
Pandas = DF3[DF3['age-months']>85]
print(Pandas)
'''

# EXAMPLE: 3
'''
Pandas = DF3[DF3['age-months']<28]
print(Pandas)
'''

# EXAMPLE: 4
'''
Pandas = DF3[(DF3['age-months']>28) & (DF3['age-months']<80)]
print(Pandas)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# INDEXING

# EXAMPLE: 1
'''
Pandas = DF1.set_index('Name')
print(Pandas)
'''

# EXAMPLE: 2
'''
print(Pandas.reset_index())
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# LOC and ILOC FUNCTION
'''
- The Loc and iLoc functions in Pandas are used to slice a dataset and filter row wise data.
- loc selects rows and columns with specific labels and iloc selects rows and columns at specific integer positions.
'''

# LOC
'''
loc -> location -> loc() is label based data selecting method which means that we have to pass the name of the 
row or column which we want to select
'''

# EXAMPLE: 1
'''
Pandas = DF3.loc[2]               # -> only 2nd row
print(Pandas)
'''
# EXAMPLE: 2
'''
Pandas = DF3.loc[2:]             # -> from 2nd row to all
print(Pandas)
'''
# EXAMPLE: 3
'''
Pandas = DF3.loc[:5]             # -> upto 5th row
print(Pandas)
'''
# EXAMPLE: 4
'''
Pandas = DF3.loc[2:4]            # -> [including:including] -> all data
print(Pandas)
'''
# EXAMPLE: 5
'''
Pandas = DF3.loc['age-months']
print(Pandas)                                   # -> Error
'''
# EXAMPLE: 6
'''
Pandas = DF3.loc[2:4,'age-months']            # -> specified('First_Name')
print(Pandas)
'''
# EXAMPLE: 7
'''
Pandas = DF3.loc[2:4,'age-months':'weight-lbs']      # -> [including:including, 'including':'including'] -> upto
print(Pandas)
'''
# EXAMPLE: 8
'''
Pandas = DF3.loc[2:4,['age-months','weight-lbs']]    # -> [including:including, 'including':'including'] -> specified
print(Pandas)
'''
# EXAMPLE: 9
'''
Pandas = DF3.loc[2:4, :]                      # -> specified rows and all columns
print(Pandas)
'''
# EXAMPLE: 10
'''
Pandas = DF3.loc[:,'age-months':'weight-lbs']        # -> specified columns and all rows
print(Pandas)
'''
# EXAMPLE: 11
'''
Pandas = DF3.loc[:,:]                         # -> all data
print(Pandas)
'''
# EXAMPLE: 12
'''
Pandas = DF3.loc[:,'age-months':'weight-lbs':2]      # -> column slicing
print(Pandas)
'''
# EXAMPLE: 13
'''
Pandas = DF3.loc[0:10:2,:]                     # -> row slicing
print(Pandas)
'''

# -------------------------------------------------------------------------------------------------------------------- #

# ILOC
'''
iloc -> integer location -> iloc() is a indexed based selecting method which means that we have to pass integer index 
in the method to select specific row/column.
'''

# EXAMPLE: 1
'''
Pandas = DF3.iloc[2]              # -> only specified(2)
print(Pandas)
'''
# EXAMPLE: 2
'''
Pandas = DF3.iloc[10:]            # -> from row 10 onwards
print(Pandas)
'''
# EXAMPLE: 3
'''
Pandas = DF3.iloc[:5]            # -> [including(0):excluding(5)] -> all column data
print(Pandas)
'''
# EXAMPLE: 4
'''
Pandas = DF3.iloc[2:5]           # -> [including:excluding] -> all column data
print(Pandas)
'''
# EXAMPLE: 5
'''
Pandas = DF3.iloc[0:2,0:2]       # -> [including:excluding,including:excluding] -> only specified data
print(Pandas)
'''
# EXAMPLE: 6
'''
Pandas = DF3.iloc[0:0,0:1]       # -> try with 1 at stop position(row and column)
print(Pandas)
'''
# EXAMPLE: 7
'''
Pandas = DF3.iloc[2:4,[1,3]]     # -> [including:including, ['including','including']] -> specified
print(Pandas)
'''
# EXAMPLE: 8
'''
Pandas = DF3.iloc[2,4]           # -> fetched row and column simultaneosly
print(Pandas)
print(DF3)
'''
# EXAMPLE: 9
'''
Pandas = DF3.iloc[1:2, :]        # -> [including:excluding, all column data]
print(Pandas)
'''
# EXAMPLE: 10
'''
Pandas = DF3.iloc[:,1:2]         # -> [all row data, including:excluding]
print(Pandas)
'''
# EXAMPLE: 11
'''
Pandas = DF3.iloc[:,:]          # -> all data
print(Pandas)
'''
# EXAMPLE: 12
'''
Pandas = DF3.iloc[:,0:5:2]      # -> column slicing
print(Pandas)
'''
# EXAMPLE: 13
'''
Pandas = DF3.iloc[0:5:2,:]      # -> row slicing
print(Pandas)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# HEAD
'''
- Outputs the first five rows of your DataFrame by default.
- We could also pass a number as well: .head(10) would output the top ten rows, for example.
'''

# EXAMPLE: 1
'''
H = DF3.head()
print(H)
'''
# EXAMPLE: 2
'''
H = DF3.head(10)
print(H)
'''

# -------------------------------------------------------------------------------------------------------------------- #

# TAIL
'''
- To see the last five rows use .tail() by default
- tail(2) also accepts a number, and in this case we printing the bottom two rows, for example.
'''

# EXAMPLE: 1
'''
T = DF3.tail()
print(T)
'''

# EXAMPLE: 2
'''
T = DF3.tail(2)
print(T)
'''

# -------------------------------------------------------------------------------------------------------------------- #

# INFO
'''
- Gives info of our loaded data.
- .info() provides the essential details about your dataset, such as the number of rows and columns, the number of 
  non-null values, what type of data is in each column, and how much memory your DataFrame is using.
'''

# EXAMPLE: 1
'''
I = DF3.info()
print(I)
'''
# EXAMPLE: 2
'''
I = DF3.info()
print(I)
'''

# -------------------------------------------------------------------------------------------------------------------- #

# DESCRIBE
'''
Using describe() on an entire DataFrame we can get a summary of the distribution of continuous variables
'''

# EXAMPLE:
'''
D = DF3.describe()
print(D)
'''

# -------------------------------------------------------------------------------------------------------------------- #

# VALUE COUNTS
'''
- Gives the count of repeated element.
'''

# EXAMPLE: 1
'''
V = DF3['age-months'].value_counts()
print(V)
'''
# EXAMPLE: 2
'''
V = DF3['Group'].value_counts()
print(V)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# HANDLING DUPLICATES
'''
H = DF2.append(DF2)
print(H)
print(H.shape)

HD = H.drop_duplicates()
print(HD)
print(HD.shape)
'''

# GET DUMMIES
'''
G = pd.get_dummies(DF1['Name'])
print(G)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# GROUP BY
'''
It allows you to split your data into separate groups to perform computation for better analysis.
'''

# EXAMPLE: 1
'''
DF5 = pd.read_csv('nba.csv')

GB = DF5.groupby('Team')

obj = GB.first()                            # -> Team column will show as first column and in alphabetical order as well as unique values only
print(obj)

obj1 = GB.get_group('Atlanta Hawks')        # -> fetch all Atlanta Hawks as Team
print(obj1)
'''

# EXAMPLE: 2
'''
DF5 = pd.read_csv('nba.csv')

GB = DF5.groupby(['Team', 'Position'])

obj = GB.first()                            # -> Team column will show as first column and in alphabetical order as well as unique values only
print(obj)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# JOIN
'''
- join() is used to combine two DataFrames on the basis of their indexes but not on columns
- {‘inner’, ‘outer’}, default ‘outer’. Outer for UNION and Inner for INTERSECTION. 
'''

#------------------------------------------------------------------------------------------#

DF6 = pd.DataFrame({'A':['A1', 'A2', 'A3', 'A4', 'A5'],
                    'B':['B1', 'B2', 'B3', 'B4', 'B5'],
                    'C':['C1', 'C2', 'C3', 'C4', 'C5']},
                    index=[1, 2, 3, 4, 5])

DF7 = pd.DataFrame({'D':['D1', 'D2', 'D3', 'D4', 'D5'],
                    'E':['E1', 'E2', 'E3', 'E4', 'E5'],
                    'F':['F1', 'F2', 'F3', 'F4', 'F5']},
                    index=[4, 5, 6, 7, 8])

#------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
J1 = pd.concat([DF6, DF7], axis=1)                    # -> by default join = 'outer', outer means UNION operation
print(J1)
'''
# EXAMPLE: 2
'''
J2 = pd.concat([DF6, DF7], axis=1, join='inner')      # -> join = 'inner', inner means INTERSECTION operation
print(J2)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# MERGE
'''
- merge() is primarily used to specify the columns you wanted to join on.
'''

#------------------------------------------------------------------------------------------#

DF8 = pd.DataFrame({'Sr':[1, 2, 3, 4],
                    'A':['A1', 'A2', 'A3', 'A4']})

DF9 = pd.DataFrame({'Sr':[1, 2, 5, 6],
                    'B':['B1', 'B2', 'B3', 'B4']})

#------------------------------------------------------------------------------------------#

# EXAMPLE: 1
'''
M = pd.merge(DF8, DF9, on='Sr', how='inner')
print(M)
'''
# EXAMPLE: 2
'''
M1 = pd.merge(DF8, DF9, on='Sr', how='left')
print(M1)
'''
# EXAMPLE: 3
'''
M2 = pd.merge(DF8, DF9, on='Sr', how='right')
print(M2)
'''
# EXAMPLE: 4
'''
M3 = pd.merge(DF8, DF9, on='Sr', how='outer')
print(M3)
'''
# EXAMPLE: 5
'''
M4 = pd.merge(DF8, DF9, left_index=True, right_index=True)
print(M4)
'''

#------------------------------------------------------------------------------------------#

DF10 = pd.DataFrame({"Car": ['BMW', 'Lexus', 'Audi', 'Mustang', 'Bentley', 'Jaguar'],
                           "Units": [100, 150, 110, 80, 110, 90]})

DF11 = pd.DataFrame({"Car": ['BMW', 'Lexus', 'Tesla', 'Mustang', 'Mercedes', 'Jaguar'],
                           "Reg_Price": [7000, 1500, 5000, 8000, 9000, 6000]})

#------------------------------------------------------------------------------------------#

# EXAMPLE: 6
'''
M5 = pd.merge(DF10, DF11, how ="left", indicator=True)
print(M5)
'''
# EXAMPLE: 7
'''
M6 = pd.merge(DF10, DF11, how ="right", indicator=True)
print(M6)
'''
# EXAMPLE: 8
'''
M7 = pd.merge(DF10, DF11, how ="outer", indicator=True)
print(M7)
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# DATA PREPROCESSING
'''
- Data preprocessing is the first machine learning step
- Data Preprocessing is the process of doing a pre-analysis of data, in order to transform them into a standard and 
  normalized format.
- Some specified Machine Learning model needs information in a specified format. For example, Random Forest algorithm 
  does not support null values.
'''
# STEP INVOLVED IN DATA PRE-PROCESSING
'''
 1. Data Cleaning
 2. Data Transformation
 3. Data Reduction

1. Data Cleaning: 
- The data can have many irrelevant and missing parts. To handle this part, data cleaning is done. It involves handling 
  of missing data, noisy data etc. 
  
 (a). Missing Data: 
        This situation arises when some data is missing in the data. It can be handled in various ways. Some of them are:

 (b). Noisy Data: 
        Noisy data is a meaningless data that can’t be interpreted by machines. It can be generated due to faulty data 
        collection, data entry errors etc. It can be handled in following ways :  
        
2. Data Transformation: 
- This step is taken in order to transform the data in appropriate suitable forms. This involves following ways: 

 (a). Normalization: 
        It is done in order to scale the data values in a specified range (-1.0 to 1.0 or 0.0 to 1.0) 
 
 (b). Attribute Selection: 
        In this strategy, new attributes are constructed from the given set of attributes to help the process. 

3. Data Reduction: 
- Since data mining is a technique that is used to handle huge amount of data. While working with huge volume of data, 
  analysis became harder in such cases. In order to get rid of this, we uses data reduction technique. It aims to 
  increase the storage efficiency and reduce data storage and analysis costs. 
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# HANDLING NULL VALUES

# EXAMPLE:
'''
N = DF1.isnull()
print(N)

N1 = DF1.isnull().sum()            # -> sum() for no. of null in each column
print(N1)

N2 = DF1.isnull().sum().sum()      # -> to get total null values count
print(N2)

N3 = DF1.dropna()                   # removes the row which have atleast 1 null value
print(N3)

N4 = DF1.dropna(axis=1)             # axis=1 for column, removes the column which have atleast 1 null value
print(N4)

N5 = DF1.fillna(0)                                 # fill all NaN with 0
print(N5)

N6 = DF1.fillna(method='ffill')                    # use previous data to fill next NaN data
print(N6)

N7 = DF1.fillna(method='ffill', limit=1)           # for next value only
print(N7)

N8 = DF1.fillna(method='bfill')                    # use next data to fill previous NaN data
print(N8)

FILL = {"Name":1, "Age":2, "City":3, "Salary":4}
N9 = DF1.fillna(value=FILL)
print(N9)

N10 = DF1['Age'] = DF1['Age'].fillna(DF1['Age'].mean())
print(N10)
'''

# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻

# SCALING or FEATURE SCALING
'''
- In Data Processing, we try to change the data in such a way that the model can process it without any problems. And 
  Feature Scaling is one such process in which we transform the data into a better version. Feature Scaling is done to 
  normalize the features in the dataset into a finite range.
- There are several ways to do feature scaling. 5 most commonly used feature scaling techniques:
    1. Absolute Maximum Scaling
    2. Min-Max Scaling
    3. Normalization
    4. Standardization
    5. Robust Scaling
'''

data = pd.read_csv('scaling.csv')
df = pd.DataFrame(data)

# 1. ABSOLUTE MAXIMUM SCALING
'''
- Find the absolute maximum value of the feature in the dataset
- Divide all the values in the column by that maximum value
- Dataset will lie between 0 and 1
'''

# EXAMPLE:
'''
AMS = df['Age'] / df['Age'].max()
print(AMS)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 2. MIN-MAX SCALING
'''
- In min-max you will subtract the minimum value in the dataset with all the values and then divide this by the range of 
  the dataset(maximum-minimum).
- In this case, our dataset will lie between 0 and 1
- X(new) = ( X - X(min) ) / ( X(max) - X(min) )
'''

# EXAMPLE:
'''
MMS = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
print(MMS)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 3. NORMALIZATION
'''
- Instead of using the min() value in the previous case, in this case, we will be using the average() value.
- In scaling, you are changing the range of your data while in normalization you are changing the shape of the 
  distribution of your data.
- Dataset will lie between -1 to 1
- X(new) = [ X - X(mean) ] / [ X(max) - X(min) ]
'''

# EXAMPLE:
'''
NS = (df['Age'] - df['Age'].mean()) / (df['Age'].max() - df['Age'].min())
print(NS)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 4. STANDARDIZATION
'''
- In standardization, we calculate the z-score (value) for each of the data points and replaces those with these values.
- This will make sure that all the features are centred around the mean value with a standard deviation value of 1
- mean = 0 and variance = 1
- X(new) = [ X - X(mean) ] / [ std() ]
'''

# EXAMPLE:
'''
SD = (df['Age'] - df['Age'].mean()) / df['Age'].std()
print(SD)
'''

# ○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○○

# 5. ROBUST SCALING
'''
- In this method, we need to subtract median value from all the data points and then divide it by the Inter Quartile 
  Range(IQR) value.
- This method centres the median value at zero and this method is robust to outliers.
- X(new) = [ X - X(median) ] / IQR
'''

# EXAMPLE:
'''
RS = (df['Age']- df['Age'].median()) / [(df['Age'].quantile(0.25)) - (df['Age'].quantile(0.75))]
print(RS)
'''
# ☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻☺☻