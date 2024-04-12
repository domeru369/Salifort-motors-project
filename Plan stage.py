
#  Step 1. Imports
#   Import packages
#   Load dataset

# Import packages
### YOUR CODE HERE ### 
# Import packages for data manipulation
import pandas as pd
import numpy as np
# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
# Import packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample
# Import packages for data modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Load dataset into a dataframe
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
df0.head()


Step 2. Data Exploration (Initial EDA and data cleaning)

# Gather basic information about the data
df0.info()

# Gather descriptive statistics about the data
df0.describe(include = 'all')

# Display all column names
pd.set_option('display.max_columns', None)
columns_list = df0.columns.tolist()
print(columns_list)

# Rename columns as needed
df0 = df0.rename(columns = {'average_montly_hours': 'average_monthly_hours', 'Department': 'department', 'Work_accident': 'work_accident', 'time_spend_company': 'Tenure'})

# Display all column names after the update
columns_list = df0.columns.tolist()
print(columns_list)

# Check for missing values
df0.isna().sum()

# Check for any duplicate entries in the data.
df0.duplicated().sum()

# Inspect some rows containing duplicates as needed
df0[df0.duplicated()]

# Drop duplicates and save resulting dataframe in a new variable as needed
df = df0.drop_duplicates()

# Display first few rows of new dataframe as needed
df.duplicated().sum()
df.head()

#check the data types of all variables
df.dtypes


# Check for outliers in the data.
# Create a boxplot to visualize distribution of `tenure` and detect any outliers
plt.figure(figsize=(4,2))
sns.boxplot(x=df['Tenure'])
plt.plot()


# Handling outliers with percentile and quantile and reassigning the outlier values
percentile25 = df['Tenure'].quantile(0.25)
percentile75 = df['Tenure'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
df.loc[df['Tenure'] > upper_limit, 'Tenure'] = upper_limit


# Get numbers of people who left vs. stayed
print(df1['left'].value_counts())
print()
# Get percentages of people who left vs. stayed
df['left'].value_counts(normalize=True)

# the dataset is imbalanced in terms of people that left and people that did not leave. people that left took up 17% of the whole dataset

