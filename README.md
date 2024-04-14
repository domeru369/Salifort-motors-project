# Salifort-motors-project
This capstone project is from the advanced google data analytics certificate. It is an opportunity for me to analyze a dataset and build predictive models that can provide insights to the Human Resources (HR) department of a large consulting firm. 

# Project workflow
We would be incoporating the 6 data analysis steps into the PACE project workflow.
* P - Plan (Ask, Prepare and Process)
* A - Analyze (Analyze)
* C - Construct - build the model 
* E - Execute (Share and Act)

# 1. PLAN
## Understand the business scenario and problem
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to me as a data analytics professional and ask me to provide data-driven suggestions based on my understanding of the data. 

## Ask
They have the following question: what’s likely to make the employee leave the company?
My goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.
If I can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.


## Prepare
The dataset that I will be using in this lab contains 15,000 rows and 10 columns for the variables listed below. 

For more information about the data, refer to its source on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv).

Variable  |Description |
-----|-----|
satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
last_evaluation|Score of employee's last performance review [0&ndash;1]|
number_project|Number of projects employee contributes to|
average_monthly_hours|Average number of hours employee worked per month|
time_spend_company|How long the employee has been with the company (years)
Work_accident|Whether or not the employee experienced an accident while at work
left|Whether or not the employee left the company
promotion_last_5years|Whether or not the employee was promoted in the last 5 years
Department|The employee's department
salary|The employee's salary (U.S. dollars)

# Limitations of the dataset
* The data was last updated 3 years ago (2021)
* The salary variable is lacks details. It is grouped categorically (low, medium and high)

## Process
* The first stage of the project workflow is the [Plan stage.](https://github.com/domeru369/Salifort-motors-project/blob/main/Plan%20stage.py)
* I chose Python to perform analysis because python boasts a rich ecosystem of libraries specifically designed for machine learning tasks. Popular libraries like NumPy, Pandas and Scikit-learn. It also has libraries for visualizations such as matplotlib and seaborn.
* I imported all the packages needed for cleaning and manipulating data
* I loaded the data set into a dataframe and loaded it
* I displayed the first few rows of the data and then gathered descriptive analysis of the data.
* I gathered basic information about the data then checking if all the data types are correct and there are no inncorrect data types
* I renamed the column names for more better understanding
* There are no missing entries in the data
* 3,008 rows contain duplicates. That is 20% of the data and I dropped them as needed.
* I checked for outliers in all the variables and discovered that the tenure variable contained outliers so I handled them by reassigning them, that is, changing the values to ones that fit within the general distribution of the dataset. I created a floor and ceiling at a quantile eg 75% and 25% percentile. Any value above the 75% mark or below the 25% mark are changed to fit within the walls of the IQR(Inter Quantile Range). 
* The dataset is imbalanced in terms of people that left and people that stayed. People that left took up 17% of the whole dataset and people that stayed took up to 83%.

## Analyze
* The second stage of the project workflow is the [Analyze stage.](https://github.com/domeru369/Salifort-motors-project/blob/main/Analyze%20stage.py)
* This is where I examineD variables that i was interested in, and created plots to visualize relationships between variables in the data.
* 
