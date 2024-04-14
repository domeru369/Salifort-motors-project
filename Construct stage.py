# Step 3. Model Building, Step 4. Results and Evaluation
#LOGISTIC REGRESSION
# copy the dataframe to a new one
X = df.copy()
# Isolate the independent variables by dropping the dependent varialble 'left'
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

# Isolate target variable (dependent variable)
y = df['left']
y.head()

#confirm the data types of the independent varaibles
X.dtypes


#split the data
# Split dataset into training and holdout datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, stratify=y, random_state=0)


#get shape of the data
X_train.shape, X_test.shape, y_train.shape, y_test.shape


#build the model
model = LogisticRegression(penalty='none', max_iter=500)
model.fit(X_train, y_train)


#call the model.coef to output all the coefficients
pd.Series(model.coef_[0], index=X.columns)


#call the model interpret
model.intercept_


# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
training_probabilities


# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit_data = X_train.copy()
# 2. Create a new `logit` column in the `logit_data` df
logit_data['logit'] = [np.log(prob[1] / prob[0]) for prob in training_probabilities]

# plot the logit
sns.regplot(x='average_monthly_hours', y='logit', data=logit_data, scatter_kws={'s': 2,'alpha': 0.5})
plt.title('Log-odds: average_monthly_hours');

#plot the logit
sns.regplot(x='satisfaction_level', y='logit', data=logit_data, scatter_kws={'s': 2,'alpha': 0.5})
plt.title('Log-odds: satisfaction_level');


#using the model to predict on the test data (unseen data to understand how the data will perfrom on unseen data or data it has not experienced)
y_preds = model.predict(X_test)


#Score the model (accuracy) on the test data
model.score(X_test, y_test)

#plot the confusion matrix
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=['stayed', 'left'],
)
disp.plot();


# Create a classification report
target_labels = ['stayed', 'left']
print(classification_report(y_test, y_preds, target_names=target_labels))


#Create a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))
# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1],reverse=True)
feature_importance

# Plot the feature importances
import seaborn as sns
sns.barplot(x=[x[1] for x in feature_importance],
y=[x[0] for x in feature_importance],
orient='h')
plt.title('Feature importance');




## TREE BASED MODEL
# Important imports for modeling and evaluation
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, roc_curve, auc


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


#print the size of the train and test dataset
for x in [X_train, X_test]:
    print(len(x))


#build the decision tree model
decision_tree = DecisionTreeClassifier(random_state=0)

#fit the model
decision_tree.fit(X_train, y_train)

#use the model to predict on the test data
dt_pred = decision_tree.predict(X_test)


#evaluate the model by attaining important metrics
print("Decision Tree")
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, dt_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, dt_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, dt_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, dt_pred))
print("auc:", "%.6f" % metrics.roc_auc_score(y_test, dt_pred))


#plot the confusion matrix
cm = metrics.confusion_matrix(y_test, dt_pred, labels = decision_tree.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = decision_tree.classes_)
disp.plot()


#plot the decision tree
plt.figure(figsize=(20,12))
plot_tree(decision_tree, max_depth=2, fontsize=14, feature_names=X.columns);


#plot the feature importances
importances = decision_tree.feature_importances_
forest_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax);


#tune the parameters because decision trees are prone to overfitting

tree_para = {'max_depth':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,30,40,50],
             'min_samples_leaf': [2,3,4,5,6,7,8,9, 10, 15, 20, 50]}

scoring = {'accuracy', 'precision', 'recall', 'f1', 'roc_auc'}

                   

#check combination of values
tuned_decision_tree = DecisionTreeClassifier(random_state=0)

clf = GridSearchCV(tuned_decision_tree, 
                   tree_para, 
                   scoring = scoring, 
                   cv=5, 
                   refit="f1")

clf.fit(X_train, y_train)


#the best estimatos of the model
clf.best_estimator_

#the best validation score which is the f1 score
print("Best Avg. Validation Score: ", "%.4f" % clf.best_score_)


#determine the best scores combination 
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




## RANDOM FOREST
#import all necessary packages
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

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


#split the data 
# 3. Split into train and test sets
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y,
                                              test_size=0.2, random_state=42)

# 4. Split into train and validate sets
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr,
                                                  test_size=0.25, random_state=42)



#print the size of the train, validation and test dataset
for x in [X_train, X_val, X_test]:
    print(len(x))
  


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



#fit the model
rf_cv.fit(X_train, y_train)

#the best score which is the F1
rf_cv.best_score_

#the best parameters
rf_cv.best_params_


#determine the best scores combination 
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
results = make_results('RF cv', rf_cv, 'f1')
results

#pickle the model in order to save the model so we do not build the model all over again when we want to use it again
import pickle
path = '/home/jovyan/work/'

# Pickle the model
with open(path + 'rf_cv_model.pickle', 'wb') as to_write:
    pickle.dump(rf_cv, to_write) 


# Open pickled model
with open(path+'rf_cv_model.pickle', 'rb') as to_read:
    rf_cv = pickle.load(to_read)





## XGBOOST MODEL

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


#fit the model and the time it will take the model to build
get_ipython().run_cell_magic('time', '', 'xgb_cv.fit(X_train, y_train)')

#where the model will be saved
path = '/home/jovyan/work/'

# Pickle the model
with open(path + 'xgb_cv_model.pickle', 'wb') as to_write:
    pickle.dump(xgb_cv, to_write) 


# Open pickled model
with open(path+'xgb_cv_model.pickle', 'rb') as to_read:
    xgb_cv = pickle.load(to_read)


#the best score which is the f1
xgb_cv.best_score_


#the best parameters
xgb_cv.best_params_



# Call 'make_results()' on the GridSearch object
xgb_cv_results = make_results('XGB cv', xgb_cv, 'f1')
results = pd.concat([results, xgb_cv_results], axis=0)
results


# Use random forest model to predict on validation data
rf_val_preds = rf_cv.best_estimator_.predict(X_val)


#determine the best scores combination 
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




# Get validation scores for RF model
rf_val_scores = get_test_scores('RF val', rf_val_preds, y_val)

# Append to the results table
results = pd.concat([results, rf_val_scores], axis=0)
results


cm = confusion_matrix(y_val, rf_val_preds, labels=rf_cv.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['stayed', 'left'])
disp.plot();



# Use XGBoost model to predict on validation data
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)

# Get validation scores for XGBoost model
xgb_val_scores = get_test_scores('XGB val', xgb_val_preds, y_val)

# Append to the results table
results = pd.concat([results, xgb_val_scores], axis=0)
results


#Use XGBoost model to predict on test data
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)

# Get test scores for XGBoost model
xgb_test_scores = get_test_scores('XGB test', xgb_test_preds, y_test)

# Append to the results table
results = pd.concat([results, xgb_test_scores], axis=0)
results


# In[107]:


#Use random forest model to predict on test data
rf_test_preds = rf_cv.best_estimator_.predict(X_test)

# Get test scores for random forest model
rf_test_scores = get_test_scores('rf test', rf_test_preds, y_test)

# Append to the results table
results = pd.concat([results, rf_test_scores], axis=0)
results



#confusion matrix
cm = confusion_matrix(y_test, rf_test_preds, labels=rf_cv.classes_)
# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['stayed', 'left'])
disp.plot();



#plot feature importance
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()

