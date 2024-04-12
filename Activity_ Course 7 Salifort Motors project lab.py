
# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# ### Import packages

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


# Step 2. Data Exploration (Initial EDA and data cleaning)

# Gather basic information about the data
df0.info()

# Gather descriptive statistics about the data
df0.describe(include = 'all')


# ### Rename columns
# Display all column names
### YOUR CODE HERE ###
pd.set_option('display.max_columns', None)
columns_list = df0.columns.tolist()
print(columns_list)

#rename columns
df0 = df0.rename(columns = {'average_montly_hours': 'average_monthly_hours', 'Department': 'department', 'Work_accident': 'work_accident', 'time_spend_company': 'Tenure'})

# Display all column names after the update
columns_list = df0.columns.tolist()
print(columns_list)


# Check missing values

# Check for any missing values in the data.

# In[6]:


# Check for missing values
### YOUR CODE HERE ###
df0.isna().sum()


# ### Check duplicates

# Check for any duplicate entries in the data.

# In[7]:


# Check for duplicates
### YOUR CODE HERE ###
df0.duplicated().sum()


# In[8]:


# Inspect some rows containing duplicates as needed
### YOUR CODE HERE ###
df0[df0.duplicated()]


# In[7]:


# Drop duplicates and save resulting dataframe in a new variable as needed
### YOUR CODE HERE ###
df = df0.drop_duplicates()

# Display first few rows of new dataframe as needed
### YOUR CODE HERE ###
df.duplicated().sum()


# In[24]:


df.head()


# In[25]:


df.dtypes


# ### Check outliers

# Check for outliers in the data.

# In[27]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
### YOUR CODE HERE ###
plt.figure(figsize=(4,2))
sns.boxplot(x=df['Tenure'])
plt.plot()


# In[11]:


# Determine the number of rows containing outliers
### YOUR CODE HERE ###
percentile25 = df['Tenure'].quantile(0.25)
percentile75 = df['Tenure'].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr
df.loc[df['Tenure'] > upper_limit, 'Tenure'] = upper_limit


# Certain types of models are more sensitive to outliers than others. When you get to the stage of building your model, consider whether to remove outliers, based on the type of model you decide to use.

# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables)
# 
# 

# 💭
# ### Reflect on these questions as you complete the analyze stage.
# 
# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# [Double-click to enter your responses here.]

# ## Step 2. Data Exploration (Continue EDA)
# 
# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# In[19]:


# Get numbers of people who left vs. stayed
### YOUR CODE HERE ###
df['left'].value_counts(normalize=True)
# Get percentages of people who left vs. stayed
### YOUR CODE HERE ###


# the dataset is imbalanced in terms of people that left and people that did not leave. people that left took up 17% of the whole dataset

# ### Data visualizations

# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# In[104]:


#histogram of satisfaction level 
plt.figure(figsize=(6,4))
sns.histplot(df['satisfaction_level'], binrange = (0.1,1))
plt.title('Histogram of satisfaction level')


# The satisfaction level data is uniform indicating that all data points are evenly distributed.

# In[124]:


# Create a plot as needed 
### YOUR CODE HERE ###

# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.legend(labels=['left', 'stayed'])
plt.title('Monthly hours by satisfaction level', fontsize='14');


# - There are 3 groups of employees that left. The first group, employees that their satisfaction level were nearly 0 worked for long hours (average of 240 to 315 hours monthly). It is quite obvious why their satisfaction level was very low. 
# - The second group, employees whose satisfaction levels were about 0.4 and worked less hours (130 to 160 monthly) and their reasons for leaving are less clear.
# - The third group, some employees with relatively high satisfaction levels and moderate working hours (210-280 monthly) also left.It is important to understand why they left.

# In[102]:


#last evaluation histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='last_evaluation',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('last evaluation histogram');


# - The last evaluation for people that left is bimodal meaning it has two distinct peaks, indicating that the data has two modes while the last evaluation for people that stayed is left skewed.

# In[123]:


# Create a plot as needed 
### YOUR CODE HERE ###

# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.legend(labels=['left', 'stayed'])
plt.title('Monthly hours by last evaluation', fontsize='14');


# - There are 2 groups of employees that left by last evaluation, the first group worked around 130 to 160 hours but did not perform very well. 
# - The second group who performed excellently but were overworked.
# - Working long hours does not guarantee a good evaluation score.

# In[100]:


#histogram of promotion  
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='promotion_last_5years',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Promotion last 5 years histogram');


# In[134]:


# Set figure and axes
plt.figure(figsize =(8,5))

# Create boxplot 
sns.boxplot(data=df, x='average_monthly_hours', y='promotion_last_5years', hue='left', orient="h")
tenure_stay = df[df['left']==0]['Tenure']
tenure_left = df[df['left']==1]['Tenure']
plt.title('Monthly hours by Promotion', fontsize='14')

plt.show();


# - The boxplot above shows that very few people who were promoted in the last 5 years left.
# - All employees who were promoted and worked less hours left. 
# - All employees who were not promoted and worked the most(240+) hours left.
# - All employees who were promoted worked average hours(155-240)
# 

# In[96]:


#number of project histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='number_project',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Number of project histogram');


# - Employees that stayed are more than people that left in regards to number of project 
# - The highest number of projects most employees work on are 3 and 4 but the highest number of people that left worked on 2 projects

# In[95]:


# Create a plot as needed 
### YOUR CODE HERE ###
plt.figure(figsize = (8,6))
# Set figure and axes
# Create boxplot 
sns.boxplot(data=df,  x='average_monthly_hours', y='number_project', hue='left',orient="h")
plt.title('Monthly hours by number of projects', fontsize='14')
plt.show()


# There are two groups of employees that left the company;
# - Group A: Those who worked on fewer projects and clocked considerably less than the average monthly hours (around 150 hours).
# - Group B: They were involved in a higher number of projects and worked significantly longer hours, exceeding 250 hours per month.
# - Those in Group A might have left due to factors like underutilization, lack of challenge, or they were fired.
# - Group B's heavy workload and long hours indicate a potential risk of burnout. They might have left due to excessive pressure, stress, or feeling overwhelmed.

# In[94]:


#number of project histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='Tenure',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Tenure histogram');


# - The highest number of employees that stayed and left are in their third tenure
# - There are few longer tenured employees, this could be high ranking employees with high salaries too.

# In[8]:


# Create a plot as needed 
### YOUR CODE HERE ###

# Set figure and axes
plt.figure(figsize =(10,10))

# Create boxplot
sns.boxplot(data=df, x='satisfaction_level', y='Tenure', hue='left', orient="h")
plt.title('Satisfaction by tenure', fontsize='14')

plt.show();


# There are many observations in this plot;
# - Dissatisfied New Hires: the highest number of employees that left fall into the category of dissatisfied individuals with relatively shorter tenures. 
# - A small group of 4 years tenured employees with seemingly low satisfaction levels also left. 
# - Retention of Long-Term Employees: The plot shows that most employees with the longest tenures did not leave. Their satisfaction levels appear similar to those of newer employees who chose to stay.

# In[92]:


plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='salary',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by salary histogram');


# - Employees with low salaries left the most.
# - There are few employees with high salaries.

# In[91]:


# Create a plot as needed 
### YOUR CODE HERE ###
plt.figure(figsize = (6,5))
# Set figure and axes
# Create boxplot 
sns.boxplot(data=df,  x='average_monthly_hours', y='salary', hue='left',orient="h")
plt.title('Monthly hours by salary', fontsize='14')
plt.show()


# - Employees with low, medium and high salaries that worked 150 to 260 hours on average all left.

# In[85]:


plt.figure(figsize=(9,5))
sns.histplot(data=df,
             x='department',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by department histogram');


# - People in sales department left the most followed by the technical department. 

# In[88]:


plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='work_accident',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by work accident histogram');


# In[ ]:


- A small amount of people left by work accident


# In[87]:


# Create a plot as needed
### YOUR CODE HERE ###
# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap indicates many low correlated variables',
          fontsize=18)
plt.show();


# - The heatmap shows that none of the variables has high multicollinearity

# ### Insights

# It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave.

# # paCe: Construct Stage
# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# 🔎
# ## Recall model assumptions
# 
# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the constructing stage.
# 
# - Do you notice anything odd?
# - Which independent variables did you choose for the model and why?
# - Are each of the assumptions met?
# - How well does your model fit the data?
# - Can you improve it? Is there anything you would change about the model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# [Double-click to enter your responses here.]

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# [Double-click to enter your responses here.]
# categorical prediction task - to find out why employees are leaving and to predict whether they will leave or not

# ### Identify the types of models most appropriate for this task.

# [Double-click to enter your responses here.]
# logistic regression and random forest model and xgboost model, i will cross validate too

# ### Modeling
# 
# Add as many cells as you need to conduct the modeling process.

# In[31]:


### YOUR CODE HERE ###
X = df.copy()
# Drop unnecessary columns
X = X.drop(['left'], axis=1)
# Encode the `salary` column as an ordinal numeric category
X['salary'] = (
    X['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
X = pd.get_dummies(X, drop_first=False)

# Display the new dataframe
X.head()


# In[32]:


# Isolate target variable
y = df['left']
y.head()


# In[33]:


X.dtypes


# In[34]:


#split your data
# Split dataset into training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y, random_state=0)


# In[35]:


#get shape of the data
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[36]:


#build the model
model = LogisticRegression(penalty='none', max_iter=500)
model.fit(X_train, y_train)


# In[37]:


#call the model.coef to output all the coefficients
pd.Series(model.coef_[0], index=X.columns)


# # Interpretation 
# Based on the logistic regression model, a one unit increase in satisfaction level is associated with a 3.8 decrease in the log odds of employees leaving.
# To interpret this in a more explanatory way, we will get the exponention of -3.87 and use 1 to minus it eg e-3.87 = 0.02, 1-0.02 = 50 = 0.5%  
# we can then say, for every one unit increase in satisfaction level, we wxpect that the odds the person will leave decreases by 0.5%
# 2) for every one unit increase in the evaluation score, we expect that the odds the person will leave increases by 52%
# 3) for every one unit increase in the average monthly hours, we expect that the odds the person will leave increases by 100.37%
# 4) for every one unit increase in tenure, we expect that the odds the person will leave increases by 182% 

# In[38]:


model.intercept_


# In[39]:


# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
training_probabilities


# In[40]:


# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit_data = X_train.copy()
# 2. Create a new `logit` column in the `logit_data` df
logit_data['logit'] = [np.log(prob[1] / prob[0]) for prob in training_probabilities]


# In[41]:


sns.regplot(x='average_monthly_hours', y='logit', data=logit_data, scatter_kws={'s': 2,'alpha': 0.5})
plt.title('Log-odds: average_monthly_hours');


# In[42]:


sns.regplot(x='satisfaction_level', y='logit', data=logit_data, scatter_kws={'s': 2,'alpha': 0.5})
plt.title('Log-odds: satisfaction_level');


# In[43]:


#use your model to predict on the test data (unseen data to understand how the data will perfrom on unseen data or data it hasnt experienced)
y_preds = model.predict(X_test)


# In[44]:


#Score the model (accuracy) on the test data
model.score(X_test, y_test)


# In[45]:


cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=['stayed', 'left'],
)
disp.plot();


# the true negatives are 720, that means the model predicted accurately that 720 did not leave the company .
# the true positives are 74, which means the model accurately predicted that 74 people left the company.
# the false positives are 240 which means 240 people did not leave the company but the model inaccurately predicted that they left. 
# the false negatives are 800, which means 800 employees actually left the company but the model inaccurately predicted that they did not leave. 
# the model is weak at predicting true positives 

# In[46]:


# Create a classification report
target_labels = ['stayed', 'left']
print(classification_report(y_test, y_preds, target_names=target_labels))


# The model is 82% accurate. the precision score is 39% meaning that the model was poor at predicting false positives. the model was also bad at predicting false negatives as the recall score was 16%, the harmonic mean of the precision and recall score is 22%

# In[47]:


#Create a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))
# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1],reverse=True)
feature_importance


# In the logistic regression, tenure was the most important feature that was used in prediction followed by last evaluation and the departments

# In[48]:


# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance],
y=[x[0] for x in feature_importance],
orient='h')
plt.title('Feature importance');


# 

# # TREE BASED MODEL

# In[64]:


### YOUR CODE HERE ###
# Important imports for modeling and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc


# In[50]:


#isolated x and y variables
y = df['left']
#copy the data 
X = df.copy()
#isolate x varaibles
X = X.drop("left", axis = 1)
#encode salary feature as an ordinal catergorical variable
X['salary'] = (
    X['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
X = pd.get_dummies(X, drop_first=False)

#split the data in train and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[51]:


#print the size of the train and test dataset
for x in [X_train, X_test]:
    print(len(x))


# In[52]:


#build the decision tree model
decision_tree = DecisionTreeClassifier(random_state=0)

decision_tree.fit(X_train, y_train)

dt_pred = decision_tree.predict(X_test)


# In[66]:


#evaluate the model by attaining important metrics
print("Decision Tree")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred))
print("auc:", "%.6f" % metrics.roc_auc_score(y_test, dt_pred))


# The model is 96% accurate. the precision score is 87% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 91%, the harmonic mean of the precision and recall score is 89%. The overall performance of the model is satisfactory but we have to build other powerful models for more accuracy

# In[54]:


#plot the confusion matrix
cm = metrics.confusion_matrix(y_test, dt_pred, labels = decision_tree.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = decision_tree.classes_)
disp.plot()


# the true negatives of the decision model are 2,500, that means the model predicted accurately that 2,500 did not leave the company . the true positives are 430, which means the model accurately predicted that 430 people left the company. the false positives are 62 which means 62 people did not leave the company but the model inaccurately predicted that they left. the false negatives are 39, which means 39 employees actually left the company but the model inaccurately predicted that they did not leave. 

# In[55]:


plt.figure(figsize=(20,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns);


#  The first line of information in each node is the feature and split point that the model identified as being most predictive. In other words, this is the question that is being asked at that split. For our root node, the question was: Is the customer satisfactory level less than or equal to 0.46?
# 
# At each node, if the answer to the question it asks is "yes," the sample would move to the child node on the left. If the answer is "no," the sample would go to the child node on the right.
# 
# gini refers to the node's Gini impurity. This is a way of measuring how "pure" a node is. The value can range from 0 to 0.5. A Gini score of 0 means there is no impurity—the node is a leaf, and all of its samples are of a single class. A score of 0.5 means the classes are all equally represented in that node.
# 
# samples is simply how many samples are in that node, and value indicates how many of each class are in the node. Returning to the root node, we have value = [7474, 1519]. Notice that these numbers sum to 8,993, which is the number of samples in the node. This tells us that 7,474 employees in this node did not leave (y=0) and 1,519 employees left (y=1).
# 
# Lastly, we have class. This tells us the majority class of the samples in each node.
# 
# This plot tells us that, if we could only do a single split on a single variable, the one that would most help us predict whether an employee will leave is their satisfactory level.
# 
# If we look at the nodes at depth one, we notice that the number of projects and tenure also are both strong predictors (relative to the features we have) of whether or not employees will leave.
# 
# This is a good indication that it might be worthwhile to return to your EDA and examine these features more closely.
# 
# Finally, it's worth noting that there is no single question that can be asked—for any feature—that would cause a majority of samples in one of the child nodes to be of class "left." The tree must get to depth two (i.e., two questions must be asked) before this happens.

# In[56]:


importances = decision_tree.feature_importances_

forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax);


# In[57]:


#tune the parameters because decision trees are prone to overfitting

tree_para = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
             'min_samples_leaf': [2,3,4,5,6,7,8,9, 10, 15, 20, 50]}

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}


# In[58]:


#check combination of values
tuned_decision_tree = DecisionTreeClassifier(random_state=0)

clf = GridSearchCV(tuned_decision_tree, 
                   tree_para, 
                   scoring = scoring, 
                   cv=5, 
                   refit="f1")

clf.fit(X_train, y_train)


# In[106]:


clf.best_estimator_


# In[59]:


print("Best Avg. Validation Score: ", "%.4f" % clf.best_score_)


# In[73]:


#determine the best scores combination 
### YOUR CODE HERE ###

results = pd.DataFrame(columns=['Model', 'F1', 'Recall', 'Precision', 'Accuracy','auc'])

def make_results(model_name, model_object):
    """
    Accepts as arguments a model name (your choice - string) and
    a fit GridSearchCV model object.

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean F1 score across all validation folds.  
    """

    # Get all the results from the CV and put them in a df.
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(mean f1 score).
    best_estimator_results = cv_results.iloc[cv_results['mean_test_f1'].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row.
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    auc = best_estimator_results.mean_test_roc_auc

 
    # Create table of results
    table = pd.DataFrame({'Model': [model_name],
                          'F1': [f1],
                          'Recall': [recall],
                          'Precision': [precision],
                          'Accuracy': [accuracy],
                          'auc': [auc]
                         }
                        )

    return table

result_table = make_results("Tuned Decision Tree", clf)

result_table


# The model after tuning the hyperparameters is 97% accurate. the precision score is 96% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 91%, the harmonic mean of the precision and recall score is 93%. The auc score is 96.1 meaning that the model's predictions are 96.1% correct. The overall performance of the model is satisfactory but we have to build other powerful models for more accuracy

# # Random Forest Model and XGBoost

# Ethical considerations 
# 1. What are you being asked to do?
# 
# Business need and modeling objective
# 
# Currently, there is a high rate of turnover among Salifort employees. (Note: In this context, turnover data includes both employees who choose to quit their job and employees who are let go). Salifort’s senior leadership team is concerned about how many employees are leaving the company. Salifort strives to create a corporate culture that supports employee success and professional development. Further, the high turnover rate is costly in the financial sense. Salifort makes a big investment in recruiting, training, and upskilling its employees. 
# 
# If Salifort could predict whether an employee will leave the company, and discover the reasons behind their departure, they could better understand the problem and develop a solution. 
# A machine learning model that predicts whether an employee will leave the company based on their job title, department, number of projects, average monthly hours, and any other relevant data points. A good model will help the company increase retention and job satisfaction for current employees, and save money and time training new employees. 
# 
# Modeling design and target variable
# 
# The data dictionary shows that there is a column called left. This is a binary value that indicates whether an employee left the company or not. This will be the target variable. In other words, for employee, the model should predict whether an employee left or not.
# 
# This is a classification task because the model is predicting a binary class.
# 
# Select an evaluation metric
# 
# To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:
# 
# False positives: When the model predicts an employee left when in fact the employee stayed
# False negatives: When the model predicts an employee stayed when in fact the employee left
# 2. What are the ethical implications of building the model?
# False positives are worse for the company, because Time and money spent trying to retain employees who were likely to stay anyway could be better invested in other areas and Employees may feel pressured or micromanaged if they are wrongly identified as flight risks.
# 
# False negatives are worse for the company too, because it can result in higher turnover rates, impacting productivity, knowledge retention, and recruitment costs and if signs of dissatisfaction are missed, you lose the chance to address concerns and potentially prevent departures.
# 
# 
# The stakes are relatively even. You want to help reduce turnover rates and time and money spent trying to retain customers 
# F1 score is the metric that places equal weight on true postives and false positives, and so therefore on precision and recall.
# 
# 
# 3. How would you proceed?
# 
# Modeling workflow and model selection process
# 
# Previous work with this data has revealed that there are ~10,000 employees in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
# 
# Split the data into train/validation/test sets (60/20/20)
# Fit models and tune hyperparameters on the training set
# Perform final model selection on the validation set
# Assess the champion model's performance on the test set

# In[74]:


#import all necessary packages
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


# In[75]:


#isolated x and y variables
y = df['left']
#copy the data
X = df.copy()
# Drop unnecessary columns
X = X.drop(['left'], axis=1)
# Encode the `salary` column as an ordinal numeric category
X['salary'] = (
    X['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
X = pd.get_dummies(X, drop_first=False)

# Display the new dataframe
X.head()


# In[76]:


#split the data 
# 3. Split into train and test sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y,
                                              test_size=0.2, random_state=42)

# 4. Split into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr,
                                                  test_size=0.25, random_state=42)


# In[77]:


#print the size of the train, validation and test dataset
for x in [X_train, X_val, X_test]:
    print(len(x))


# In[78]:


# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [None],
             'max_features': [1.0],
             'max_samples': [0.7],
             'min_samples_leaf': [2],
             'min_samples_split': [2],
             'n_estimators': [300],
             }

# 3. Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# 4. Instantiate the GridSearchCV object
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='f1')


# 

# In[81]:


#fit the model
rf_cv.fit(X_train, y_train)


# In[82]:


rf_cv.best_score_


# In[83]:


rf_cv.best_params_


# In[86]:


def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
        model_name (string): what you want the model to be called in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy',
                   'auc': 'mean_test_roc_auc'
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy
    auc = best_estimator_results.mean_test_roc_auc

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                          },
                         )

    return table


# In[87]:


results = make_results('RF cv', rf_cv, 'f1')
results


# The random forest model is 98% accurate. the precision score is 97% meaning that the model is good at predicting false positives. the model is also good at predicting false negatives as the recall score was 90%, the harmonic mean of the precision and recall score is 94%. The auc score is 97.6 meaning that the model's predictions are 97.6% correct.The overall performance of the model is satisfactory.

# In[88]:


import pickle
path = '/home/jovyan/work/'


# In[89]:


# Pickle the model
with open(path + 'rf_cv_model.pickle', 'wb') as to_write:
    pickle.dump(rf_cv, to_write) 


# In[90]:


# Open pickled model
with open(path+'rf_cv_model.pickle', 'rb') as to_read:
    rf_cv = pickle.load(to_read)


# In[ ]:





# # XGBoost model

# In[91]:


# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [6, 12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300]
             }

# 3. Define a dictionary of scoring metrics to capture
scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

# 4. Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='f1')


# In[92]:


get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train, y_train)')


# In[93]:


path = '/home/jovyan/work/'


# In[94]:


# Pickle the model
with open(path + 'xgb_cv_model.pickle', 'wb') as to_write:
    pickle.dump(xgb_cv, to_write) 


# In[95]:


# Open pickled model
with open(path+'xgb_cv_model.pickle', 'rb') as to_read:
    xgb_cv = pickle.load(to_read)


# In[96]:


xgb_cv.best_score_


# In[97]:


xgb_cv.best_params_


# In[98]:


# Call 'make_results()' on the GridSearch object
xgb_cv_results = make_results('XGB cv', xgb_cv, 'f1')
results = pd.concat([results, xgb_cv_results], axis=0)
results


# The xgboost model is 98% accurate. the precision score is 97.17% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 90.79%, the harmonic mean of the precision and recall score is 93%. The auc score is 97.9 meaning that the model's predictions are 97.9% correct.The overall performance of the model is satisfactory but the rf f1 score is better

# In[102]:


# Use random forest model to predict on validation data
rf_val_preds = rf_cv.best_estimator_.predict(X_val)


# In[100]:


def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): Your choice: how the model will be named in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)
    auc = roc_auc_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                          })

    return table


# In[103]:


# Get validation scores for RF model
rf_val_scores = get_test_scores('RF val', rf_val_preds, y_val)

# Append to the results table
results = pd.concat([results, rf_val_scores], axis=0)
results


# In[104]:


cm = confusion_matrix(y_val, rf_val_preds, labels=rf_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['stayed', 'left'])
disp.plot();


# In[ ]:





# In[105]:


# Use XGBoost model to predict on validation data
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)

# Get validation scores for XGBoost model
xgb_val_scores = get_test_scores('XGB val', xgb_val_preds, y_val)

# Append to the results table
results = pd.concat([results, xgb_val_scores], axis=0)
results


# In[106]:


#Use XGBoost model to predict on test data
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)

# Get test scores for XGBoost model
xgb_test_scores = get_test_scores('XGB test', xgb_test_preds, y_test)

# Append to the results table
results = pd.concat([results, xgb_test_scores], axis=0)
results


# In[107]:


#Use XGBoost model to predict on test data
rf_test_preds = rf_cv.best_estimator_.predict(X_test)

# Get test scores for XGBoost model
rf_test_scores = get_test_scores('rf test', rf_test_preds, y_test)

# Append to the results table
results = pd.concat([results, rf_test_scores], axis=0)
results


# After using both models to predict on the test data, the random forest model emerged as the champion model because it has a stronger f1 score and recall, precision and accuracy scores. The random forest model is 98.41% accurate. the precision score is 97.61% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 92.71%, the harmonic mean of the precision and recall score is 95.1%. The auc score is 96.1 meaning that the model's predictions are 96.1% correct.

# In[108]:


cm = confusion_matrix(y_test, rf_test_preds, labels=rf_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['stayed', 'left'])
disp.plot();


# the true negatives of the decision model are 2000, that means the model predicted accurately that 2000 did not leave the company . the true positives are 370, which means the model accurately predicted that 370 people left the company. the false positives are 9 which means 9 people did not leave the company but the model inaccurately predicted that they left. the false negatives are 29, which means 29 employees actually left the company but the model inaccurately predicted that they did not leave.

# In[109]:


importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()


# In the random forest model, satisfaction level was the most important feature that was used in prediction followed by last evaluation, number of projects, average monthly hours and tenure

# In[ ]:





# In[ ]:





# In[ ]:





# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# ✏
# ## Recall evaluation metrics
# 
# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.
# 
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the executing stage.
# 
# - What key insights emerged from your model(s)?
# - What business recommendations do you propose based on the models built?
# - What potential recommendations would you make to your manager/company?
# - Do you think your model could be improved? Why or why not? How?
# - Given what you know about the data and the models you were using, what other questions could you address for the team?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# # INSIGHTS
# In the random forest model, satisfaction level was the most important feature that was used in prediction followed by last evaluation, number of projects, average monthly hours and tenure
# 
# 
# 

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders
# 
# 
# 

# ### Summary of model results
# 
# - Logistic Regression
# The logistic regression model achieved precision of 39%, recall of 16%, f1-score of 22% (all weighted averages), and accuracy of 82%, on the test set. 
# 
# - Tree-based Machine Learning
# After conducting feature engineering, the decision tree model achieved AUC of 96.1%, precision of 96%, recall of 91%, f1-score of 93%, and accuracy of 97%, on the test set. 
# 
# - Random Forest Machine Learning
# The model is 98.41% accurate. the precision score is 97.61% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 92.71%, the harmonic mean of the precision and recall score is 95.1%. The auc score is 96.1 meaning that the model's predictions are 96.1% correct. The random forest model emerged as the champion model.
# 
# - Xgboost Machine Learning
# The xgboost model is 98% accurate. the precision score is 97.17% meaning that the model was good at predicting false positives. the model was also good at predicting false negatives as the recall score was 90.79%, the harmonic mean of the precision and recall score is 93%. The overall performance of the model is satisfactory but the rf f1 score is better

# ###  Recommendations
# The models and the feature importances extracted from the models confirm that employees at the company are overworked. 
# To retain employees, the following recommendations could be presented to the stakeholders:
# - Investigate reasons behind such long hours for dissatisfied employees. Are they struggling to meet unrealistic deadlines? Are they under-resourced? Implement strategies to reduce workload or improve efficiency for these employees.
# 
# - Conduct exit interviews or surveys to understand why employees with moderate satisfaction and less monthly hours left. Explore factors like lack of growth opportunities, recognition issues, or company culture.
# 
# - Conduct stay interviews with satisfied employees who are at risk of leaving (e.g., those with high satisfaction but moderate hours). This might reveal underlying concerns or areas for improvement before they leave.
# 
# - For low performers, address performance issues through targeted training or mentorship. For high performers working long hours, investigate workload distribution and consider strategies like project delegation or hiring additional staff to prevent burnout.
# 
# - Focus on identifying and promoting high-performing employees with the potential to boost retention.
# 
# - Analyze project allocation to ensure employees are challenged but not overloaded. Implement strategies like job rotation or skill development opportunities to prevent boredom or stagnation for those with fewer projects.
# 
# - Develop targeted strategies for different tenure groups. Implement strong onboarding programs for new hires to improve engagement and satisfaction. For employees in the third or fourth year, explore opportunities for growth or skill development to prevent stagnation. Foster a positive work environment that retains long-tenured employees.
# 
# - Conduct a salary review to ensure competitive compensation across all positions. Consider merit-based raises or bonuses to recognize high performance and incentivize retention.
# 
# - Investigate reasons for high turnover in specific departments. Conduct surveys or focus groups to understand departmental challenges and develop strategies to address them.
# 
# - While accidents don't seem to be a major factor, prioritize safety protocols and employee well-being to prevent future incidents.
# 
# ### NEXT STEPS
# Deeper Dives:
# 
# - Analyze Specific Departments: Focus on departments with high turnover (Sales and Technical) to understand their unique challenges. Conduct targeted surveys or focus groups within these departments to gather more specific information.
# - Performance Reviews: Investigate the relationship between performance reviews and satisfaction levels. Are there biases in the evaluation process? Do low performers receive adequate feedback and support for improvement?
# - Compensation Analysis: Conduct a more detailed analysis of salaries, bonuses, and benefits. Are there any pay gaps based on gender, race, or experience? Do the benefits packages address employee needs and preferences?
# 
# 
# 

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
