#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 00:32:54 2023

@author: jakubsz
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def display_confusion_matrix(model, X_val, y_val, clasificator_name):
    '''
    Function displays confusion matrix using sklearn.
    
    '''
    predictions = model.predict(X_val)
    cm = confusion_matrix(y_val, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=model.classes_)
    disp.plot()
    plt.title(clasificator_name)
    plt.show()

def display_roc_curve(model, X, y, title):
    '''
    Function displays ROC curve.
    
    Remember to use 'plt.figure(figsize=(8, 8))' and 'plt.show()' after using
    this function.
    
    '''
    y_pred = model.predict(X)
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)


    # Plot ROC curve

    plt.plot(fpr, tpr, lw=2, label=f'ROC curve for {title} (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for all the models')
    plt.legend(loc='lower right')

def display_accuracy_stats(model, X_t, y_t, X_v, y_v, title):
    '''
    Function displays overall accuracy values on training and validation sets.
    
    '''
    pred_train = model.predict(X_t)
    pred_valid = model.predict(X_v)

    accuracy_model_train = accuracy_score(y_t, pred_train)
    accuracy_model_val = accuracy_score(y_v, pred_valid)

    print(f'\n{title} Model:')
    print(f'Accuracy of {title} on train:      {accuracy_model_train:.4f}')
    print(f'Accuracy of {title} on validation: {accuracy_model_val:.4f}')

    return [accuracy_model_train, accuracy_model_val]

def augment_data(X, y, label_to_augment=1, target_multiplier=30):
    '''
    Function that performs data augumentaion.

    Parameters
    ----------
    X : feature dataset.
    y : target dataset.
    label_to_augment : The underrepresented class. The default is 1.
    target_multiplier : multiplier of number of samples of underrepresented
    class. The default is 30.

    Returns
    -------
    X_augmented : feature set after data augumentation.
    y_augmented : target set after data augumentation

    '''
    
    # Identify samples with the specified label
    samples_to_augment = X[y == label_to_augment]
    
    # Randomly duplicate samples
    duplicated_samples = np.random.choice(samples_to_augment.index,
                                          size=target_multiplier * len(samples_to_augment)
                                          )
    duplicated_data = X.loc[duplicated_samples]
    duplicated_labels = y.loc[duplicated_samples]
    
    # Add random noise to the duplicated samples
    noise = np.random.normal(0, 0.01, size=duplicated_data.shape)
    augmented_data = duplicated_data + noise
    
    # Concatenate the original training set and the augmented samples
    X_augmented = pd.concat([X, augmented_data])
    y_augmented = pd.concat([y, duplicated_labels])
    
    return X_augmented, y_augmented

#------------------------------------------------------------------------------
# Loading and splitting of data

DATA_AGUMENTATION = 1

# Load data
data = pd.read_csv('prepared_data.csv')

# Split data into features (X) and target (y)
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Split data into training set and a temporary set
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    random_state=42
                                                    )

# Split the temporary set into testing and validation sets
X_test, X_validate, y_test, y_validate = train_test_split(X_temp,
                                                          y_temp,
                                                          test_size=0.5,
                                                          random_state=42
                                                          )

'''
if DATA_AGUMENTATION:
    X_train, y_train = augment_data(X_train, y_train)
    
'''
classificators = pd.DataFrame(data={'Classificator:': [],
                                    'Accuracy on train': [],
                                    'Accuracy on val': []
                                    })



# Create pipelines for each model (exept the neural net)
mlp_pipeline = Pipeline([('scaler', StandardScaler()),
                         ('mlp', MLPClassifier())
                        ])


svm_pipeline = Pipeline([('preprocessor', StandardScaler()),
                         ('classifier', SVC())
                         ])

knn_pipeline = Pipeline([('preprocessor', StandardScaler()),
                         ('classifier', KNeighborsClassifier())
                         ])

dt_pipeline = Pipeline([('preprocessor', StandardScaler()),
                        ('classifier', DecisionTreeClassifier())
                        ])
rf_pipeline = Pipeline([('scaler', StandardScaler()),
                        ('classifier', RandomForestClassifier())
                        ])

#------------------------------------------------------------------------------
# Neural Network Model
'''
parameter_space_mlp = {'mlp__hidden_layer_sizes': [(10, 5), (20, 10), (5,)],
                       'mlp__activation': ['relu', 'tanh', 'logistic'],
                       'mlp__solver': ['adam', 'sgd'],
                       'mlp__max_iter': [150, 200, 250]
                       }

grid_search_mlp = GridSearchCV(mlp_pipeline,
                               param_grid=parameter_space_mlp,
                               cv=5,
                               scoring=make_scorer(matthews_corrcoef),
                               n_jobs=-1
                               )

grid_search_mlp.fit(X_train, y_train)

best_params_mlp = grid_search_mlp.best_params_
best_model_mlp = grid_search_mlp.best_estimator_

# Display results
score = display_accuracy_stats(best_model_mlp,
                               X_train, y_train,
                               X_validate, y_validate,
                               'MLP'
                               )

# Confusion matrix display
display_confusion_matrix(best_model_mlp, X_validate, y_validate, 'MLP')

#saving data to classificators DataFrame
new_row_clf = {'Classificator:': 'Neural network',
               'Accuracy on train': score[0],
               'Accuracy on val': score[1]
               }
classificators.loc[len(classificators)] = new_row_clf
'''
#------------------------------------------------------------------------------
# SVM Model
parameter_space_svm = {'classifier__C': [0.1, 1, 10],
                       'classifier__kernel': ['linear', 'rbf', 'poly'],
                       'classifier__gamma': ['scale', 'auto']
                       }

grid_search_svm = GridSearchCV(svm_pipeline,
                               param_grid=parameter_space_svm,
                               cv=4,
                               scoring=make_scorer(matthews_corrcoef),
                               n_jobs=-1
                               )

grid_search_svm.fit(X_train, y_train)

best_params_svm= grid_search_svm.best_params_
best_model_svm = grid_search_svm.best_estimator_

# Display results
score = display_accuracy_stats(best_model_svm,
                               X_train, y_train,
                               X_validate, y_validate,
                               'SVM'
                               )

# Confusion matrix display
display_confusion_matrix(best_model_svm, X_validate, y_validate, 'SVM')

#saving data to classificators DataFrame
new_row_clf = {'Classificator:': 'SVM',
               'Accuracy on train': score[0],
               'Accuracy on val': score[1]
               }
classificators.loc[len(classificators)] = new_row_clf
'''
#------------------------------------------------------------------------------
# kNN Model
parameter_space_knn = {'classifier__n_neighbors': list(range(1, 20))}

grid_search_knn = GridSearchCV(knn_pipeline,
                               param_grid=parameter_space_knn,
                               scoring=make_scorer(matthews_corrcoef),
                               n_jobs=-1,
                               cv=4
                               )
grid_search_knn.fit(X_train, y_train)

best_params_knn= grid_search_knn.best_params_
best_model_knn = grid_search_knn.best_estimator_

# Display results
score = display_accuracy_stats(best_model_knn,
                               X_train, y_train,
                               X_validate, y_validate,
                               'kNN'
                               )

# Confusion matrix display
display_confusion_matrix(grid_search_knn, X_validate, y_validate, 'kNN')

#saving data to classificators DataFrame
new_row_clf = {'Classificator:': 'kNN',
               'Accuracy on train': score[0],
               'Accuracy on val': score[1]
               }
classificators.loc[len(classificators)] = new_row_clf

#------------------------------------------------------------------------------
# Decision Tree Model
parameter_space_dt = {'classifier__criterion': ['gini', 'entropy'],
                      'classifier__splitter': ['best', 'random'],
                      'classifier__max_depth': [None, 5, 8, 10, 12, 15],
                      'classifier__min_samples_split': [2, 5, 10],
                      'classifier__min_samples_leaf': [1, 2, 4, 6, 8, 10]}

grid_search_dt = GridSearchCV(dt_pipeline,
                              param_grid=parameter_space_dt,
                              cv=4,
                              scoring=make_scorer(matthews_corrcoef),
                              )
grid_search_dt.fit(X_train, y_train)

best_params_dt= grid_search_dt.best_params_
best_model_dt = grid_search_dt.best_estimator_

# Display results
score = display_accuracy_stats(best_model_dt,
                               X_train, y_train,
                               X_validate, y_validate,
                               'Decision Tree'
                               )

# Confusion matrix display
display_confusion_matrix(best_model_dt, X_validate, y_validate, 'Decision Tree')

#saving data to classificators DataFrame
new_row_clf = {'Classificator:': 'Decision Tree',
               'Accuracy on train': score[0],
               'Accuracy on val': score[1]
               }
classificators.loc[len(classificators)] = new_row_clf

#------------------------------------------------------------------------------
# Random Forest Model

parameter_space_rf = {'classifier__n_estimators': [10, 20, 30, 50],
                      'classifier__max_depth': [None, 10, 15, 20],
                      'classifier__min_samples_split': [2, 5, 10],
                      'classifier__min_samples_leaf': [1, 2, 4]
                      }

grid_search_rf = GridSearchCV(rf_pipeline,
                              param_grid=parameter_space_rf,
                              cv=5,
                              scoring=make_scorer(matthews_corrcoef),
                              n_jobs=-1
                              )

grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
best_model_rf = grid_search_rf.best_estimator_


# Display results
score = display_accuracy_stats(best_model_rf,
                               X_train, y_train,
                               X_validate, y_validate,
                               'Random Forest'
                               )

#print("Best Parameters for Random Forest:")
#print(best_params_rf)

display_confusion_matrix(best_model_rf, X_validate, y_validate, 'Random Forest')

#saving data to classificators DataFrame
new_row_clf = {'Classificator:': 'Random Forest',
               'Accuracy on train': score[0],
               'Accuracy on val': score[1]
               }
classificators.loc[len(classificators)] = new_row_clf

#------------------------------------------------------------------------------
# Displaying the results
classificators.sort_values(by=['Accuracy on val'],
                           inplace=True,
                           ignore_index=True,
                           ascending=False
                           )
classificators.set_index('Classificator:', inplace=True)
print('\n')
print(classificators)
#print(classificators.to_latex())

#------------------------------------------------------------------------------
# ROC curve

fx_roc = plt.figure(figsize=(8, 8))
display_roc_curve(best_model_mlp, X_validate, y_validate, 'MLP')
display_roc_curve(best_model_svm, X_validate, y_validate, 'SVM')
display_roc_curve(best_model_knn, X_validate, y_validate, 'kNN')
display_roc_curve(best_model_dt, X_validate, y_validate, 'Decision Tree')
display_roc_curve(best_model_rf, X_validate, y_validate, 'Random Forest')
plt.grid()
plt.show()
'''
#------------------------------------------------------------------------------
# Statistical testing of the best method

# Statistical testing of the best method
NUMBER_OF_ITERATIONS = 100

specificity_values = []
sensitivity_values = []
tpf_values = []
fpf_values = []
accuracy_values = []

for iteration in range(1, NUMBER_OF_ITERATIONS + 1):
    print(f"Iteration {iteration}/{NUMBER_OF_ITERATIONS}")
    
    # NOTE!! If you use it, you have to comment it out from the beginning
    if DATA_AGUMENTATION:
        X_train_n, y_train_n = augment_data(X_train, y_train)

    grid_search_svm.fit(X_train_n, y_train_n)
    
    y_pred = grid_search_svm.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Specificity
    specificity = tn / (tn + fp)
    specificity_values.append(specificity)
    
    # Sensitivity
    sensitivity = tp / (tp + fn)
    sensitivity_values.append(sensitivity)
    
    # True Positive Fraction (TPF)
    tpf = tp / (tp + fn)
    tpf_values.append(tpf)
    
    # False Positive Fraction (FPF)
    fpf = fp / (fp + tn)
    fpf_values.append(fpf)
    
    # Overall Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)

median_specificity = np.median(specificity_values)
mean_specificity = np.mean(specificity_values)
std_dev_specificity = np.std(specificity_values)

median_sensitivity = np.median(sensitivity_values)
mean_sensitivity = np.mean(sensitivity_values)
std_dev_sensitivity = np.std(sensitivity_values)

median_tpf = np.median(tpf_values)
mean_tpf = np.mean(tpf_values)
std_dev_tpf = np.std(tpf_values)

median_fpf = np.median(fpf_values)
mean_fpf = np.mean(fpf_values)
std_dev_fpf = np.std(fpf_values)

median_accuracy = np.median(accuracy_values)
mean_accuracy = np.mean(accuracy_values)
std_dev_accuracy = np.std(accuracy_values)

# Display the results
print(f"\nMedian Specificity: {median_specificity}")
print(f"Mean Specificity:     {mean_specificity}")
print(f"Standard Deviation of Specificity: {std_dev_specificity}")

print(f"\nMedian Recall (Sensitivity): {median_sensitivity}")
print(f"Mean Recall (Sensitivity):     {mean_sensitivity}")
print(f"Standard Deviation of Recall (Sensitivity): {std_dev_sensitivity}")

print(f"\nMedian True Positive Fraction (TPF): {median_tpf}")
print(f"Mean True Positive Fraction (TPF):     {mean_tpf}")
print(f"Standard Deviation of TPF: {std_dev_tpf}")

print(f"\nMedian False Positive Fraction (FPF): {median_fpf}")
print(f"Mean False Positive Fraction (FPF):     {mean_fpf}")
print(f"Standard Deviation of FPF: {std_dev_fpf}")

print(f"\nMedian Overall Accuracy: {median_accuracy}")
print(f"Mean Overall Accuracy:     {mean_accuracy}")
print(f"Standard Deviation of Overall Accuracy: {std_dev_accuracy}")

print(f'\nData augmentation: {DATA_AGUMENTATION}')

results_df = pd.DataFrame({
    'Metric': ['Specificity', 'Sensitivity', 'True Positive Fraction', 'False Positive Fraction', 'Overall Accuracy'],
    'Median': [median_specificity, median_sensitivity, median_tpf, median_fpf, median_accuracy],
    'Mean': [mean_specificity, mean_sensitivity, mean_tpf, mean_fpf, mean_accuracy],
    'Standard Deviation': [std_dev_specificity, std_dev_sensitivity, std_dev_tpf, std_dev_fpf, std_dev_accuracy]
})
