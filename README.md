# Salifort-motors-project
This capstone project is from the advanced google data analytics certificate. It is an opportunity for me to analyze a dataset and build predictive models that can provide insights to the Human Resources (HR) department of a large consulting firm. 

Click this [link](https://www.coursera.org/learn/google-advanced-data-analytics-capstone/ungradedLab/uskOX/activity-course-7-salifort-motors-project-lab/lab?path=%2Fnotebooks%2FActivity_%2520Course%25207%2520Salifort%2520Motors%2520project%2520lab.ipynb) for the Jupyter Notebook version of this project.

# Project workflow
We would incorporate the 6 data analysis steps into the PACE project workflow.
* P - Plan (Ask, Prepare and Process)
* A - Analyze (Analyze)
* C - Construct - Build the models 
* E - Execute (Share and Act)

# 1. PLAN
The first stage of the project workflow is the [Plan stage.](https://github.com/domeru369/Salifort-motors-project/blob/main/Plan%20stage.py)
## Understand the business scenario and problem
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to me as a data analytics professional and ask me to provide data-driven suggestions based on my understanding of the data. 

## Ask
They have the following question: What’s likely to make the employee leave the company?
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
* The salary variable lacks details. It is grouped categorically (low, medium and high)

## Process
* I chose Python to perform analysis because python boasts a rich ecosystem of libraries specifically designed for machine learning tasks. Popular libraries like NumPy, Pandas and Scikit-learn. It also has libraries for visualizations such as matplotlib and seaborn.
* I imported all the packages needed for cleaning and manipulating data.
* I loaded the data set into a dataframe.
* I displayed the first few rows of the data and then gathered descriptive analysis of the data.
* There are no inncorrect data types.
* I renamed the column names for better understanding.
* There are no missing entries in the data
* 3,008 rows contain duplicates. That is 20% of the data and I dropped them as needed.
* I checked for outliers in all the variables and discovered that the tenure variable contained outliers so I handled them by reassigning them, that is, changing the values to ones that fit within the general distribution of the dataset. I created a floor and ceiling at a quantile eg 75% and 25% percentile. Any value above the 75% mark or below the 25% mark are changed to fit within the walls of the IQR(Inter Quantile Range). 
* The dataset is imbalanced in terms of people that left and people that stayed. People that left took up 17% of the whole dataset and people that stayed took up to 83%.

# 2. ANALYZE
* The second stage of the project workflow is the [Analyze stage.](https://github.com/domeru369/Salifort-motors-project/blob/main/Analyze%20stage.py)
* This is where I examined variables and created plots to visualize relationships between variables in the data.

## Satisfaction level 
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/histogram%20of%20satisfaction%20level.png)

* The satisfaction level distribution is left-skewed which has a longer tail on the left side, indicating more data points on the right side.

### Monthly hours and satisfaction level
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/monthly%20hours%20by%20satisfaction%20level.png)

There are 3 groups of employees that left;
* The first group, employees that their satisfaction level were nearly 0 worked for long hours (average of 240 to 315 hours monthly). It is quite obvious why their satisfaction level was very low.
* The second group, employees whose satisfaction levels were about 0.4 and worked less hours (130 to 160 monthly) and their reasons for leaving are less clear.
* The third group, some employees with relatively high satisfaction levels and moderate working hours (210-280 monthly) also left.It is important to understand why they left.



## Last evaluation
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/last_evaluation%20histogram.png)

* The last evaluation distribution is uniform, indicating that all data points are evenly distributed.

### Monthly hours and last evaluation  
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/monthly%20hours%20by%20evaluation.png)

* There are 2 groups of employees that left by last evaluation, the first group worked around 130 to 160 hours but did not perform very well. 
* The second group who performed excellently but were overworked.



## Promotion in the last five years
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/promotion%20histogram.png)

* A small amount of employees were promoted

### Monthly hours by promotion in the last 5 years
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/monthly%20hours%20by%20promotion.png)
* Employees who worked the most(240+) hours were not promoted.
* All employees who were promoted worked less(145-180) to average hours(155-240)

  

## Number of projects
### Histogram 
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/number%20of%20project%20histogram.png)

* Employees that stayed are more than employees that left in regards to number of projects
* The highest number of projects most employees work on are 3 and 4 
* The highest number of people that left worked on 2 projects

### Monthly hours by number of projects
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/monthly%20hours%20by%20number%20of%20project.png)

There are two groups of employees that left the company in regards to number of projects;
* Group A: Those who worked on fewer projects and clocked considerably less than the average monthly hours (around 150 hours).
* Group B: They were involved in a higher number of projects and worked significantly longer hours, exceeding 250 hours per month.
* Those in Group A might have left due to factors like underutilization, lack of challenge, or they were fired.
* Group B's heavy workload and long hours indicate a potential risk of burnout. They might have left due to excessive pressure, stress, or feeling overwhelmed.


## Tenure 
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/tenure%20histogram.png)

* The highest number of employees that stayed and left are in their third tenure
* There are few longer tenured employees, this could be high ranking employees with high salaries.

### Satisfaction level by tenure  
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/satisfaction%20by%20tenure.png)

There are many observations in this plot; 
* Dissatisfied New Hires: the highest number of employees that left fall into the category of dissatisfied individuals with relatively shorter tenures. 
* 4 years tenured employees who left had seemingly low satisfaction levels. 
* Retention of Long-Term Employees: The plot shows that most employees with the longest tenures did not leave. Their satisfaction levels appear similar to those of newer employees who chose to stay.

  

## Salary
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/salary%20histogram.png)

* Employees with low salaries left the most.
* There are few employees with high salaries.

### Average monthly hours and salary  
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/monthly%20hours%20by%20salary.png)

* The amount of hours employees work do not determine the salary category they fall into because employees with high, medium and low salaries fall in the same range of monthly hours.



## Department
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/department%20histogram.png)

* Employees in the sales department left the most followed by the technical and support department. 



## Work accident
### Histogram
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/work%20accident%20histogram.png)

* A small amount of employees left by work accident.


## Heatmap 
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/heatmap.png)

* The heatmap shows that none of the variables has high multicollinearity.
* Last evaluation, number of project, average monthly hours are positively correlated with each other. They also have postive correlation with left.
* Satisfaction level and promotion are negatively correlated with left.



  
# 3. CONSTRUCT
The third stage of the project workflow is the [Construct stage.](https://github.com/domeru369/Salifort-motors-project/blob/main/Construct%20stage.py)
## Ethical considerations
## Business need and modeling objective

Currently, there is a high rate of turnover among Salifort employees. (Note: In this context, turnover data includes both employees who choose to quit their job and employees who are let go). Salifort’s senior leadership team is concerned about how many employees are leaving the company. Salifort strives to create a corporate culture that supports employee success and professional development. Further, the high turnover rate is costly in the financial sense. Salifort makes a big investment in recruiting, training, and upskilling its employees.

If Salifort could predict whether an employee will leave the company, and discover the reasons behind their departure, they could better understand the problem and develop a solution. A machine learning model that predicts whether an employee will leave the company based on their job title, department, number of projects, average monthly hours, and any other relevant data points. A good model will help the company increase retention and job satisfaction for current employees, and save money and time training new employees.

## Modeling design and target variable

The data dictionary shows that there is a column called left. This is a binary value that indicates whether an employee left the company or not. This will be the target variable. In other words, for employee, the model should predict whether an employee left or not.
This is a classification task because the model is predicting a binary class.

## Select an evaluation metric

To determine which evaluation metric might be best, consider how the model might be wrong. There are two possibilities for bad predictions:
* False positives: When the model predicts an employee left when in fact the employee stayed
* False negatives: When the model predicts an employee stayed when in fact the employee left

## What are the ethical implications of building the model?m,
* False positives are worse for the company, because time and money spent trying to retain employees who were likely to stay anyway could be better invested in other areas and employees may feel pressured or micromanaged if they are wrongly identified as flight risks.
* False negatives are worse for the company too, because it can result in higher turnover rates, impacting productivity, knowledge retention, and recruitment costs and if signs of dissatisfaction are missed, they lose the chance to address concerns and potentially prevent departures.
* The stakes are relatively even. I want to help reduce turnover rates and time and money spent trying to retain customers. F1 score is the metric that places equal weight on true postives and false positives, and so therefore on precision and recall.

## How would I proceed?
Previous work with this data has revealed that there are ~10,000 employees in the sample. This is sufficient to conduct a rigorous model validation workflow, broken into the following steps:
* Split the data into train/validation/test sets (60/20/20) 
* Fit models and tune hyperparameters on the training set
* Perform final model selection on the validation set
* Assess the champion model's performance on the test set


# 4. EXECUTE 
## INSIGHTS
* There is a trend of high turnover, likely caused by factors like extended work hours, overloaded project assignments, and a general feeling of dissatisfaction.  Employees who put in long hours might feel discouraged if their efforts aren't recognized with promotions or positive performance reviews. This could be leading to burnout among a significant portion of the workforce.
* There seems to be a higher retention rate for employees who have been with the company for more than six years.

## Summary of model results
### Logistic Regression 
* The logistic regression model achieved precision of 39%, recall of 16%, f1-score of 22% (all weighted averages), and accuracy of 82%, on the test set.
* Tenure was the most important feature that was used in prediction followed by last evaluation and the departments.
* The performance of the model is poor overall.
  

### Decision Tree Model
* After hyperparameters tuning, the decision tree model achieved AUC of 96.1%, precision of 96%, recall of 91%, f1-score of 93%, and accuracy of 97%, on the test set. The overall performance of the model is satisfactory but I need to build other powerful models for better performance.
* Satisfaction level, last evaluation, number of project, average monthly hours and tenure are the most important features used in building this model respectively.

![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/decision%20tree%20confusion%20matrix.png)

* The true negatives of the decision model are 2,500, that means the model predicted accurately that 2,500 did not leave the company .
* The true positives are 430, which means the model accurately predicted that 430 people left the company. 
* The false positives are 62 which means 62 people did not leave the company but the model inaccurately predicted that they left.
* The false negatives are 39, which means 39 employees actually left the company but the model inaccurately predicted that they did not leave.

![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/decision%20tree.png)

* The first line of information in each node is the feature and split point that the model identified as being most predictive. In other words, this is the question that is being asked at that split. For our root node, the question was: Is the customer satisfactory level less than or equal to 0.46?
* At each node, if the answer to the question it asks is "yes," the sample would move to the child node on the left. If the answer is "no," the sample would go to the child node on the right.
* Gini refers to the node's Gini impurity. This is a way of measuring how "pure" a node is. The value can range from 0 to 0.5. A Gini score of 0 means there is no impurity—the node is a leaf, and all of its samples are of a single class. A score of 0.5 means the classes are all equally represented in that node.
* Samples is simply how many samples are in that node, and value indicates how many of each class are in the node. Returning to the root node, we have value = [7474, 1519]. Notice that these numbers sum to 8,993, which is the number of samples in the node. This tells us that 7,474 employees in this node did not leave (y=0) and 1,519 employees left (y=1).
* This plot tells us that, if we could only do a single split on a single variable, the one that would most help us predict whether an employee will leave is their satisfactory level.
* If we look at the nodes at depth one, we notice that the number of projects and tenure also are both strong predictors (relative to the features we have) of whether or not employees will leave.


### Random Forest Model
* The random forest model emerged as the CHAMPION MODEL.
* The model is 98.41% accurate.
* The precision score is 97.61% meaning that the model was good at predicting false positives.
* The model was also good at predicting false negatives as the recall score was 92.71%, the harmonic mean of the precision and recall score is 95.1%.
* The auc score is 96.1 meaning that the model's predictions are 96.1% correct.
  
![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/random%20forest%20confusion%20matrix.png)

* The true negatives of the random forest model are 2000, that means the model predicted accurately that 2000 did not leave the company .
* The true positives are 370, which means the model accurately predicted that 370 people left the company.
* The false positives are 9 which means 9 people did not leave the company but the model inaccurately predicted that they left.
* The false negatives are 29, which means 29 employees actually left the company but the model inaccurately predicted that they did not leave.
* In the random forest model, satisfaction level was the most important feature that was used in prediction followed by last evaluation, number of projects, average monthly hours and tenure.


### XGBoost Model
* The xgboost model is 98% accurate.
* The precision score is 97.17% meaning that the model was good at predicting false positives.
* The model was also good at predicting false negatives as the recall score was 90.79%.
* The harmonic mean of the precision and recall score is 93%. The overall performance of the model is satisfactory but the rf f1 score is better.

![image](https://github.com/domeru369/Salifort-motors-project/blob/main/Data%20visualizations/table.jpeg)


## Recommendations
The models and the feature importances extracted from the models confirm that employees at the company are overworked. 
To retain employees, the following recommendations could be presented to the stakeholders:
* Investigate reasons behind such long hours for dissatisfied employees. Are they struggling to meet unrealistic deadlines? Are they under-resourced? Implement strategies to reduce workload or improve efficiency for these employees.
* Conduct exit interviews or surveys to understand why employees with moderate satisfaction and less monthly hours left. Explore factors like lack of growth opportunities, recognition issues, or company culture.
* Conduct stay interviews with satisfied employees who are at risk of leaving (e.g., those with high satisfaction but moderate hours). This might reveal underlying concerns or areas for improvement before they leave.
* For low performers, address performance issues through targeted training or mentorship. For high performers working long hours, investigate workload distribution and consider strategies like project delegation or hiring additional staff to prevent burnout.
* Focus on identifying and promoting high-performing employees with the potential to boost retention.
* Analyze project allocation to ensure employees are challenged but not overloaded. Implement strategies like job rotation or skill development opportunities to prevent boredom or stagnation for those with fewer projects.
* Develop targeted strategies for different tenure groups. Implement strong onboarding programs for new hires to improve engagement and satisfaction. For employees in the third and fourth year, explore opportunities for growth or skill development to prevent stagnation. Foster a positive work environment that retains long-tenured employees.
* Conduct a salary review to ensure competitive compensation across all positions. Consider merit-based raises or bonuses to recognize high performance and incentivize retention.
* Investigate reasons for high turnover in specific departments eg sales department. Conduct surveys or focus groups to understand departmental challenges and develop strategies to address them.
* While accidents don't seem to be a major factor, prioritize safety protocols and employee well-being to prevent future incidents.

### NEXT STEPS
Deeper Dives:
- Analyze Specific Departments: Focus on departments with high turnover (Sales and Technical) to understand their unique challenges. Conduct targeted surveys or focus groups within these departments to gather more specific information.
- Performance Reviews: Investigate the relationship between performance reviews and satisfaction levels. Are there biases in the evaluation process? Do low performers receive adequate feedback and support for improvement?
- Compensation Analysis: Conduct a more detailed analysis of salaries, bonuses, and benefits. Are there any pay gaps based on gender, race, or experience? Do the benefits packages address employee needs and preferences?




