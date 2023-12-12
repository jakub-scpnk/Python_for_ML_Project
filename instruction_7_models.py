#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 23:41:07 2023

@author: jakubsz
"""

import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam

data = pd.read_csv('prepared_data.csv')

# Splitting data into features (X) and target (y)
X = data.drop('Bankrupt?', axis=1)
y = data['Bankrupt?']

# Splitting data into training set and a temporary set
X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                    y,
                                                    test_size=0.4,
                                                    random_state=42
                                                    )

# Splitting the temporary set into testing and validation sets
X_test, X_validate, y_test, y_validate = train_test_split(X_temp,
                                                          y_temp,
                                                          test_size=0.5,
                                                          random_state=42
                                                          )

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_validate_scaled = scaler.transform(X_validate)

#------------------------------------------------------------------------------
# Neural Network Model
#------------------------------------------------------------------------------
model = Sequential([Dense(10, input_dim=X_train_scaled.shape[1], activation='relu'),
                    Dense(5, activation='relu'),
                    Dense(1, activation='sigmoid')
                    ])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled,
          y_train,
          validation_data=(X_validate_scaled, y_validate),
          epochs=10,
          batch_size=10,
          verbose=0
          )

# Evaluate the model on the test and validation sets
loss, accuracy = model.evaluate(X_test_scaled, y_test)
train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train)
v_loss, v_accuracy =  model.evaluate(X_validate_scaled, y_validate)
print('\nMLP Model:')
print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
print(f'Val Loss:   {v_loss:.4f}, Val Accuracy:   {v_accuracy:.4f}')
#print(f'Test Loss:  {loss:.4f}, Test Accuracy:  {accuracy:.4f}')

#------------------------------------------------------------------------------
# SVM Model
#------------------------------------------------------------------------------
# Fit the SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train_scaled, y_train)

# Predict using the SVM model
predictions = model.predict(X_test_scaled)

# Evaluate the predictions
accuracy_train = model.score(X_train_scaled, y_train)
accuracy_val = model.score(X_validate_scaled, y_validate)
accuracy_test = model.score(X_test_scaled, y_test)
print('\nSVM Model:')
print(f'Accuracy of SVM on train:      {accuracy_train:.4f}')
print(f'Accuracy of SVM on validation: {accuracy_val:.4f}')
#print(f'Accuracy of SVM on test:       {accuracy_test:.4f}')

#------------------------------------------------------------------------------
# kNN Model
#------------------------------------------------------------------------------

neigh = KNeighborsClassifier()
parameter_space = {'n_neighbors': list(range(1,20))}
clf_knn = GridSearchCV(neigh , parameter_space , n_jobs=-1, cv=4)
clf_knn.fit(X_train_scaled, y_train)

# Evaluate the predictions
accuracy_train = clf_knn.score(X_train_scaled, y_train)
accuracy_val = clf_knn.score(X_validate_scaled, y_validate)
accuracy_test = clf_knn.score(X_test_scaled, y_test)
print('\nkNN Model:')
print(f'Accuracy of kNN  on train:      {accuracy_train:.4f}')
print(f'Accuracy of kNN  on validation: {accuracy_val:.4f}')
#print(f'Accuracy of kNN  on test:       {accuracy_test:.4f}')
print(f'Best parameter value: {clf_knn.best_params_}')

#------------------------------------------------------------------------------
# Decision Tree Model
#------------------------------------------------------------------------------

dt_classifier = DecisionTreeClassifier()
parameter_space = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [None, 5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              }

clf_dt = GridSearchCV(dt_classifier,
                      parameter_space,
                      cv=4,
                      scoring='accuracy'
                      )

clf_dt.fit(X_train_scaled, y_train)

# Evaluate the predictions
accuracy_train = clf_dt.score(X_train_scaled, y_train)
accuracy_val = clf_dt.score(X_validate_scaled, y_validate)
accuracy_test = clf_dt.score(X_test_scaled, y_test)
print('\nDT Model:')
print(f'Accuracy of DT on train:      {accuracy_train:.4f}')
print(f'Accuracy of DT on validation: {accuracy_val:.4f}')
#print(f'Accuracy of DT on test:       {accuracy_test:.4f}')
print(f'Best parameter value: {clf_dt.best_params_}')
