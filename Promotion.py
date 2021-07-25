# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## <h1><center>Predict whether the Employee of an Organization should get Promotion or Not?</center></h1>
# 
# &nbsp;
# 
# <h2>Aim:</h2>
# 
# The client is a large MNC and they have 9 broad verticals across the organisation. One of the problem your client is facing is around identifying the right people for promotion (only for manager position and below) and prepare them in time. Currently the process, they are following is:
#     * They first identify a set of employees based on recommendations/ past performance.
#     * Selected employees go through the separate training and evaluation program for each vertical. These programs are based on he required skill of each vertical
#     * At the end of the program, based on various factors such as training performance, an employee gets the promotion
# 
# ![image](https://economictimes.indiatimes.com/photo/20474308.cms)
# %% [markdown]
# ## Importing all the Required Libraries
# 
# * We Import Numpy, Pandas, Matplot, and Seaborn for Data Analysis and Visualizations
# * We import ipywidgets, Sweetviz, ppscore for Exploratory Data Analysis
# * We Import Sklearn, Imblearn for Machine Learning Modelling

# %%
# import all the required libraries

# for mathematical operations
import numpy as np

# for dataframe operations
import pandas as pd

# for data visualizations
import seaborn as sns
import matplotlib.pyplot as plt

# for machine learning
import sklearn
import imblearn

# setting up the size of the figures
plt.rcParams['figure.figsize'] = (16, 5)
# setting up the style of the plot
plt.style.use('seaborn-paper')

# %% [markdown]
# ## Reading the Dataset
# 
# * We are having two datasets, i.e., Training and Testing Datasets
# * Read both the datasets 
# * Training Datasets is used to train the Machine learning Models
# * After learning the patterns from the Testing Datasets, the Target Variable has to be predicted.

# %%
# reading the datasets

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# %% [markdown]
# ## Examining the Data
# 
# * First, we will check the shape of the dataset
# * Second, we will check the head, tail, and sample of the datasets
# * Third, we will check the Data Description
# * Then, we will check the Data Types of the columns present in the data.
# * Atlast, we will check the Target Class Balance

# %%
# lets check the shape of the train and test datasets
print("Shape of the Training Data :", train.shape)
print("Shape of the Test Data :", test.shape)


# %%
# columns in Training Data
train.columns


# %%
# columns in Test Data
test.columns


# %%
# lets check the head of the dataset
train.head()


# %%
# lets check the head of the dataset
test.head()


# %%
# check the tail of the dataset

train.tail()


# %%
# check the tail of the dataset

test.tail()

# %% [markdown]
# ## <center><b>Data Description</b></center>
# 
# <table>
#     <tr>
#         <td><b>Variable</b></td>
#         <td><b>Definition</b></td>
#     </tr>
#     <tr>
#         <td>employee_id</td>
#         <td>Unique ID<td>
#     </tr>
#     <tr>
#         <td>department</td>
#         <td>Department of employee</td>
#     </tr>
#     <tr>
#         <td>region</td>
#         <td>Region of employment (unordered)</td>
#     </tr>
#     <tr>
#         <td>education</td>
#         <td>Education Level</td>
#     </tr>
#     <tr>
#         <td>gender</td>
#         <td>Gender of Employee</td>
#     </tr>
#     <tr>
#         <td>recruitment_channel</td>
#         <td>Channel of recruitment for employee</td>
#     </tr>
#     <tr>
#         <td>no_of_trainings</td>
#         <td>no of other trainings completed in previous year on soft skills, technical skills etc.</td>
#     </tr>
#     <tr>
#         <td>age</td>
#         <td>Age of Employee</td>
#     </tr>
#     <tr>
#         <td>previous_year_rating</td>
#         <td>Employee Rating for the previous year</td>
#     </tr>
#     <tr>
#         <td>length_of_service</td>
#         <td>Length of service in years</td>
#     </tr>
#     <tr>
#         <td>KPIs_met >80%</td>
#         <td>if Percent of KPIs(Key performance Indicators) >80% then 1 else 0</td>
#     </tr>
#     <tr>
#         <td>awards_won?</td>
#         <td>if awards won during previous year then 1 else 0</td>
#     </tr>
#     <tr>
#         <td>avg_training_score</td>
#         <td>Average score in current training evaluations</td>
#     </tr>
#     <tr>
#         <td>is_promoted	(Target)</td>
#         <td>Recommended for promotion</td>
#     </tr>
# </table>

# %%
# values in Departments

train['department'].value_counts()


# %%
# values in Region

train['region'].value_counts()

# %% [markdown]
# # Discriptive Stats

# %%
# check descriptive statistics for numerical columns
train.describe().style.background_gradient(cmap = 'PuBuGn')

# %% [markdown]
# # Outliers
# 
# ### Data is quite uniform and thus has negated the outliers.

# %%
# check descriptive statistics for categorical columns
train.describe(include = 'object')


# %%
# check the Target Class Balance

plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('seaborn-whitegrid')

plt.subplot(1, 2, 1)
sns.countplot(train['is_promoted'],)

plt.xlabel('Promoted or Not?', fontsize = 10)

plt.subplot(1, 2, 2)
train['is_promoted'].value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%.2f%%', startangle = 90,
                                       labels = ['1','0'], shadow = True, pctdistance = 0.5)
plt.axis('off')

plt.suptitle('Target Class Balance', fontsize = 15)
plt.show()

# %% [markdown]
# ### Target Class is Highly Imbalanced.
# ### We must balance these classes of Target Class. 
# %% [markdown]
# # Treatment of missing values

# %%
# missing values in training data set

# calculate the total missing values in the dataset
train_total = train.isnull().sum()

# calculate the percentage of missing values in the dataset
train_percent = ((train.isnull().sum()/train.shape[0])*100).round(2)

# calculate the total missing values in the dataset
test_total = test.isnull().sum()

# calculate the percentage of missing values in the dataset
test_percent = ((test.isnull().sum()/test.shape[0])*100).round(2)

# make a dataset consisting of total no. of missing values and percentage of missing values in the dataset
train_missing_data = pd.concat([train_total, train_percent, test_total, test_percent],
                                axis=1, 
                                keys=['Train_Total', 'Train_Percent %','Test_Total', 'Test_Percent %'],
                                sort = True)

# lets check the head
train_missing_data.style.bar(color = ['black'])


# %%
# lets impute the missing values in the Training Data

train['education'] = train['education'].fillna(train['education'].mode()[0])
train['previous_year_rating'] = train['previous_year_rating'].fillna(train['previous_year_rating'].mode()[0])

# lets check whether the Null values are still present or not?
print("Number of Missing Values Left in the Training Data :", train.isnull().sum().sum())


# %%
# lets impute the missing values in the Testing Data

test['education'] = test['education'].fillna(test['education'].mode()[0])
test['previous_year_rating'] = test['previous_year_rating'].fillna(test['previous_year_rating'].mode()[0])

# lets check whether the Null values are still present or not?
print("Number of Missing Values Left in the Training Data :", test.isnull().sum().sum())

# %% [markdown]
# ### Missing data is filled respectively in the training and testing data using mod
# %% [markdown]
# # Outlier Detection
# %% [markdown]
# ## Univariate Analysis
# 

# %%
# plot pie chart for the columns where we have very few categories
plt.rcParams['figure.figsize'] = (16,5)
plt.style.use('seaborn-whitegrid')

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 1)
labels = ['0','1']
sizes = train['KPIs_met >80%'].value_counts()
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
explode = [0, 0]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('KPIs Met > 80%', fontsize = 20)

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 2)
labels = ['1', '2', '3', '4', '5']
sizes = train['previous_year_rating'].value_counts()
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
explode = [0, 0, 0, 0, 0.1]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Previous year Ratings', fontsize = 20)

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 3)
labels = ['0', '1']
sizes = train['awards_won?'].value_counts()
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
explode = [0,0.1]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Awards Won?', fontsize = 20)


plt.legend()
plt.show()

# %% [markdown]
# ### We, can see that there are some pie charts, we have for representing KPIs, Previous year Ratings, and Awards Won?
# 
# ### Also, The one Big Pattern is that only some of the employees could reach above 80% of KPIs set.
# ### Most of the Employees have a very low rating for the previous year, and
# ### very few employees, probably 2% of them could get awards for their work, which is normal.

# %%
# check the distribution of trainings undertaken by the employees

plt.rcParams['figure.figsize'] = (17, 4)
sns.countplot(train['no_of_trainings'], palette = 'winter')
plt.xlabel(' ', fontsize = 14)
plt.title('Distribution of Trainings undertaken by the Employees')
plt.show()


# %%
# check the Age of the Employees

plt.rcParams['figure.figsize'] = (8, 4)
plt.hist(train['age'], color = 'blue')
plt.title('Distribution of Age among the Employees', fontsize = 20)
plt.xlabel('Age of the Employees')
plt.grid()
plt.show()


# %%
# check different Departments

plt.rcParams['figure.figsize'] = (12, 6)
sns.countplot(y = train['department'], palette = 'cividis', orient = 'v')
plt.xlabel('')
plt.ylabel('Department Name')
plt.title('Distribution of Employees in Different Departments', fontsize = 20)
plt.grid()

plt.show()


# %%
# check distribution of different Regions

plt.rcParams['figure.figsize'] = (12,15)
plt.style.use('fivethirtyeight')
sns.countplot(y = train['region'], palette = 'inferno', orient = 'v')
plt.xlabel('')
plt.ylabel('Region')
plt.title('Different Regions', fontsize = 15)
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# %%
# plotting pie chart for the columns where we have very few categories
plt.rcParams['figure.figsize'] = (16,5)

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 1)
labels = train['education'].value_counts().index
sizes = train['education'].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0, 0, 0.1]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Education', fontsize = 20)

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 2)
labels = train['gender'].value_counts().index
sizes = train['gender'].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0, 0]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Gender', fontsize = 20)

# plotting a pie chart to represent share of Previous year Rating of the Employees
plt.subplot(1, 3, 3)
labels = train['recruitment_channel'].value_counts().index
sizes = train['recruitment_channel'].value_counts()
colors = plt.cm.copper(np.linspace(0, 1, 5))
explode = [0,0,0.1]

plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)
plt.title('Recruitment Channel', fontsize = 20)

plt.show()

# %% [markdown]
# From, the above pie charts displayed for representing Education, Gender, and Recruitment Channel.
# 
# lets infer the Main Highlights
# Very Few employees are actually working only after their Secondary Education, 
# Obviously Females are again in Minority as compared to their Male Counterparts.
# and the Recruitment Channel, says that the Referred Employees are very less, i.e., most of the employees are recruited either by sourcing, or some other recruitment agencies, sources etc.
# %% [markdown]
# # Bivariate Analysis
# 
# * First, we will perform Categorical vs Categorical Analysis using Stacked and Grouped Bar Charts with the help of crosstab function.
# * Second, we will perform Categorical vs Numerical Analysis using Bar Charts, Box plots, Strip plots, Swarm plots, Boxen plots, Violin Plots, etc
# * Atlast, we will perform Numerical vs Numerical Analysis using Scatter plots.

# %%
# Lets compare the Gender Gap in the promotion

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (13, 5)
plt.style.use('seaborn-paper')
x = pd.crosstab(train['gender'], train['is_promoted'])
colors = plt.cm.Wistia(np.linspace(0, 2, 5))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False)
plt.title('Effect of Gender on Promotion', fontsize = 17)
plt.xlabel(' ')
plt.show()

# %% [markdown]
# ### As we have already seen that the Females are in Minority, but when it comes to Promotion, they are competing with their Men Counterparts neck-to-neck. That's a great Inference.

# %%
# lets compare the effect of different Departments and Promotion

plt.rcParams['figure.figsize'] = (15,4)
plt.style.use('seaborn-paper')
x = pd.crosstab(train['department'], train['is_promoted'])
colors = plt.cm.copper(np.linspace(0, 1, 3))
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'area', stacked = False)
plt.title('Effect of Department on Promotion', fontsize = 15)
plt.xticks(rotation = 20)
plt.xlabel(' ')
plt.show()

# %% [markdown]
# ### From, the above chart we can see that almost all the Departments have a very similar effect on Promotion. So, we can consider that all the Departments have a similar effect on the promotion. Also, this column comes out to be lesser important in making a Machine Learning Model, as it does not contribute at all when it comes to Predicting whether the Employee should get Promotion.

# %%
# Effect of Age on the Promotion

plt.rcParams['figure.figsize'] = (15,4)
sns.boxenplot(train['is_promoted'], train['age'], palette = 'PuRd')
plt.title('Effect of Age on Promotion', fontsize = 15)
plt.xlabel('Is the Employee Promoted?', fontsize = 15)
plt.ylabel('Age of the Employee', fontsize = 15)
plt.show()


# %%
# Department Vs Average Training Score

plt.rcParams['figure.figsize'] = (16, 7)
sns.boxenplot(train['department'], train['avg_training_score'], palette = 'spring')
plt.title('Average Training Scores from each Department', fontsize = 15)
plt.ylabel('Promoted or not', fontsize = 10)
plt.xlabel('Departments', fontsize = 10)
plt.show()

# %% [markdown]
# # Multivariate Analysis
# 
# * First, we will use the Correlation Heatmap to check the correlation between the Numerical Columns
# * Then we will check the ppscore or the Predictive Score to check the correlation between all the columns present in the data.
# * Then, we will use Bubble Charts, split Violin plots, Hue with Bivariate Plots.
# %% [markdown]
# ## Feature Engineering
# 
# Feature engineering is the process of using domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. Feature engineering can be considered as applied machine learning itself.
# 
# * There are mutliple ways of performing feature engineering.
# * So many people in the Industry consider it the most important step to improve the Model Performance.
# * We should always understand the columns well to make some new features using the old existing features.
# * Let's discuss the ways how we can perform feature engineering
#     * We can perform Feature Engineering by Removing Unnecassary Columns
#     * We can do it by Extracting Features from the Date and Time Features.
#     * We can do it by Extracting Features from the Categorcial Features.
#     * We can do it by Binnning the Numerical and Categorical Features.
#     * We can do it by Aggregating Multiple Features together by using simple Arithmetic operations
#     
# * Here, we are only going to perform Feature Engineering by Aggregating some features together.

# %%
# lets create some extra features from existing features to improve our Model

# creating a Metric of Sum
train['sum_metric'] = train['awards_won?']+train['KPIs_met >80%'] + train['previous_year_rating']
test['sum_metric'] = test['awards_won?']+test['KPIs_met >80%'] + test['previous_year_rating']

# creating a total score column
train['total_score'] = train['avg_training_score'] * train['no_of_trainings']
test['total_score'] = test['avg_training_score'] * test['no_of_trainings']


# %%
# lets remove some of the columns which are not very useful for predicting the promotion.

# we already know that the recruitment channel is very least related to promotion of an employee, so lets remove this column
# even the region seems to contribute very less, when it comes to promotion, so lets remove it too.
# also the employee id is not useful so lets remove it.

train = train.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)
test = test.drop(['recruitment_channel', 'region', 'employee_id'], axis = 1)

# lets check the columns in train and test data set after feature engineering
train.columns


# %%
'''
lets check the no. of employee who did not get an award, did not acheive 80+ KPI, previous_year_rating as 1
and avg_training score is less than 40
but, still got promotion.
''' 

train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 
      (train['awards_won?'] == 0) & (train['avg_training_score'] < 60) & (train['is_promoted'] == 1)]


# %%
# lets remove the above two columns as they have a huge negative effect on our training data

# lets check shape of the train data before deleting two rows
print("Before Deleting the above two rows :", train.shape)

train = train.drop(train[(train['KPIs_met >80%'] == 0) & (train['previous_year_rating'] == 1.0) & 
      (train['awards_won?'] == 0) & (train['avg_training_score'] < 60) & (train['is_promoted'] == 1)].index)

# lets check the shape of the train data after deleting the two rows
print("After Deletion of the above two rows :", train.shape)

# %% [markdown]
# ## Dealing with Categorical Columns
# 
# Categorical variables are known to hide and mask lots of interesting information in a data set. It’s crucial to learn the methods of dealing with such variables. If you won’t, many a times, you’d miss out on finding the most important variables in a model. It has happened with me. Initially, I used to focus more on numerical variables. Hence, never actually got an accurate model. But, later I discovered my flaws and learnt the art of dealing with such variables.
# 
# * There are various ways to encode categorical columns into Numerical columns
# * This is an Essential Step, as we Machine Learning Models only works with Numerical Values.
# * Here, we are going to use Business Logic to encode the education column
# * Then we will use the Label Encoder, to Department and Gender Columns

# %%
## Lets check the categorical columns present in the data
train.select_dtypes('object').head()


# %%
# lets check the value counts for the education column
train['education'].value_counts()


# %%
# lets start encoding these categorical columns to convert them into numerical columns

# lets encode the education in their degree of importance 
train['education'] = train['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))
test['education'] = test['education'].replace(("Master's & above", "Bachelor's", "Below Secondary"),
                                                (3, 2, 1))

# lets use Label Encoding for Gender and Department to convert them into Numerical
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train['department'] = le.fit_transform(train['department'])
test['department'] = le.fit_transform(test['department'])
train['gender'] = le.fit_transform(train['gender'])
test['gender'] = le.fit_transform(test['gender'])

# lets check whether we still have any categorical columns left after encoding
print(train.select_dtypes('object').columns)
print(test.select_dtypes('object').columns)


# %%
# lets check the data after encoding
train.head(3)

# %% [markdown]
# ## Splitting the Data
# 
# This is one of the most Important step to perform Machine Learning Prediction on a Dataset,
# We have to separate the Target and Independent Columns.
# * We store the Target Variable in y, and then we store the rest of the columns in x, by deleting the target column from the data
# * Also, we are changing the name of test dataset to x_test for ease of understanding.

# %%
# lets split the target data from the train data

y = train['is_promoted']
x = train.drop(['is_promoted'], axis = 1)
x_test = test

# lets print the shapes of these newly formed data sets
print("Shape of the x :", x.shape)
print("Shape of the y :", y.shape)
print("Shape of the x Test :", x_test.shape)

# %% [markdown]
# ## Resampling
# 
# Resampling is the method that consists of drawing repeated samples from the original data samples. The method of Resampling is a nonparametric method of statistical inference.
# 
# * Earlier, in this Problem we noticed that the Target column is Highly Imbalanced, we need to balance the data by using some Statistical Methods.
# * There are many Statistical Methods we can use for Resampling the Data such as:
#     * Over Samping
#     * Cluster based Sampling
#     * Under Sampling.
#     
# Oversampling and undersampling in data analysis are techniques used to adjust the class distribution of a data set. These terms are used both in statistical sampling, survey design methodology and in machine learning. Oversampling and undersampling are opposite and roughly equivalent techniques
#     
# * We are going to use Over Sampling. 
# * We will not use Under Sampling to avoid data loss.

# %%
# It is very important to resample the data, as the Target class is Highly imbalanced.
# Here We are going to use Over Sampling Technique to resample the data.
# lets import the SMOTE algorithm to do the same.

from imblearn.over_sampling import SMOTE

x_resample, y_resample  = SMOTE().fit_resample(x, y.values.ravel())

# lets print the shape of x and y after resampling it
print(x_resample.shape)
print(y_resample.shape)


# %%
# lets also check the value counts of our target variable4

print("Before Resampling :")
print(y.value_counts())

print("After Resampling :")
y_resample = pd.DataFrame(y_resample)
print(y_resample[0].value_counts())


# %%
# lets create a validation set from the training data so that we can check whether the model that we have created is good enough
# lets import the train_test_split library from sklearn to do that

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)

# lets print the shapes again 
print("Shape of the x Train :", x_train.shape)
print("Shape of the y Train :", y_train.shape)
print("Shape of the x Valid :", x_valid.shape)
print("Shape of the y Valid :", y_valid.shape)
print("Shape of the x Test :", x_test.shape)

# %% [markdown]
# ## Feature Scaling
# 
# Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step
# 

# %%
# It is very import to scale all the features of the dataset into the same scale
# Here, we are going to use the standardization method, which is very commonly used.

# lets import the standard scaler library from sklearn to do that
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
x_test = sc.transform(x_test)

# %% [markdown]
# 
# ##  Machine Learning Predictive Modelling
# 
# Predictive modeling is a process that uses data and statistics to predict outcomes with data models. These models can be used to predict anything from sports outcomes and TV ratings to technological advances and corporate earnings. Predictive modeling is also often referred to as: Predictive analytics.
# %% [markdown]
# 
# 
# ### Decision Tree Classifier
# 
# A decision tree is a decision support tool that uses a tree-like model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.
# 

# %%
# Lets use Decision Trees to classify the data
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFECV

model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_valid)


# %%
from sklearn.metrics import confusion_matrix, classification_report

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_valid, y_valid))

cm = confusion_matrix(y_valid, y_pred)
plt.rcParams['figure.figsize'] = (3, 3)
sns.heatmap(cm, annot = True, cmap = 'Wistia', fmt = '.8g')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.show()


# %%
# lets take a look at the Classification Report

cr = classification_report(y_valid, y_pred)
print(cr)

# %% [markdown]
# # Realtime Prediction

# %%
train.describe()

# %% [markdown]
# # lets perform some Real time predictions on top of the Model that we just created using Decision Tree Classifier
# 
# # lets check the parameters we have in our Model
# 
#     department -> The values are from 0 to 8, (Department does not matter a lot for promotion)
#     education -> The values are from 0 to 3 where Masters-> 3, Btech -> 2, and secondary ed -> 1
#     gender -> the values are 0 for female, and 1 for male
#     no_of_trainings -> the values are from 0 to 5
#     age -> the values are from 20 to 60
#     previou_year_rating -> The values are from 1 to 5
#     length_of service -> The values are from 1 to 37
#     KPIs_met >80% -> 0 for Not Met and 1 for Met
#     awards_won> -> 0-no, and 1-yes
#     avg_training_score -> ranges from 40 to 99
#     sum_metric -> ranges from 1 to 7
#     total_score -> 40 to 710
# 

# %%
prediction = model.predict(np.array([[2,
                                      3,
                                      1,
                                      1,
                                      30,
                                      5,
                                      10,
                                      1,
                                      1,
                                      95,
                                      7,
                                      700
                                     ]]))

print("Whether the Employee should get a Promotion : 1-> Promotion, and 0-> No Promotion :", prediction)


# %%
