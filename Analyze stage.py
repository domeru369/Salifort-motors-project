
# PACE: Analyze Stage

# Data visualizations
#histogram of satisfaction level 
plt.figure(figsize=(6,4))
sns.histplot(df['satisfaction_level'], binrange = (0.1,1))
plt.title('Histogram of satisfaction level')


# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.legend(labels=['left', 'stayed'])
plt.title('Monthly hours by satisfaction level', fontsize='14');


#last evaluation histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='last_evaluation',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('last evaluation histogram');


# Create scatterplot of `average_monthly_hours` versus `last evaluation`, comparing employees who stayed versus those who left
plt.figure(figsize=(12, 9))
sns.scatterplot(data=df, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.legend(labels=['left', 'stayed'])
plt.title('Monthly hours by last evaluation', fontsize='14');


#histogram of promotion  
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='promotion_last_5years',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Promotion last 5 years histogram');


#boxplot of promotion and average monthly hours in regards to employees who left and stayed
# Set figure 
plt.figure(figsize =(8,5))
# Create boxplot 
sns.boxplot(data=df, x='average_monthly_hours', y='promotion_last_5years', hue='left', orient="h")
tenure_stay = df[df['left']==0]['Tenure']
tenure_left = df[df['left']==1]['Tenure']
plt.title('Monthly hours by Promotion', fontsize='14')
plt.show();


#number of project histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='number_project',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Number of project histogram');


#boxplot of number of project and average monthly hours in regards to employees who left and stayed
# Set figure
plt.figure(figsize = (8,6))
# Create boxplot 
sns.boxplot(data=df,  x='average_monthly_hours', y='number_project', hue='left',orient="h")
plt.title('Monthly hours by number of projects', fontsize='14')
plt.show()


#Tenure histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='Tenure',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('Tenure histogram');


# Boxplot of satisfaction level and tenure in regards to employees who left and stayed
# Set figure 
plt.figure(figsize =(10,10))
# Create boxplot
sns.boxplot(data=df, x='satisfaction_level', y='Tenure', hue='left', orient="h")
plt.title('Satisfaction by tenure', fontsize='14')
plt.show();

#salary histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='salary',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by salary histogram');


#Boxplot of salary and average monthly hours in regards to employees who left and stayed 
plt.figure(figsize = (6,5))
# Set figure and axes
# Create boxplot 
sns.boxplot(data=df,  x='average_monthly_hours', y='salary', hue='left',orient="h")
plt.title('Monthly hours by salary', fontsize='14')
plt.show()


#department histogram
plt.figure(figsize=(9,5))
sns.histplot(data=df,
             x='department',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by department histogram');


#work accident histogram
plt.figure(figsize=(6,4))
sns.histplot(data=df,
             x='work_accident',
             hue='left',
             multiple='dodge',
             shrink=0.9)
plt.title('left by work accident histogram');



# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap indicates many low correlated variables',
          fontsize=18)
plt.show();
