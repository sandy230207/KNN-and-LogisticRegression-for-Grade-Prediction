#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:29:25 2020

@author: sandy_chang
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



plt.style.use('ggplot')


# Load the dataset
df = pd.read_csv('student-mat.csv')

print(df)

# col = df.columns.tolist()

df['school'][df['school']=="GP"]=0
df['school'][df['school']=="MS"]=1

df['sex'][df['sex']=="F"]=0
df['sex'][df['sex']=="M"]=1

df['address'][df['address']=="U"]=0
df['address'][df['address']=="R"]=1

df['famsize'][df['famsize']=="GT3"]=0
df['famsize'][df['famsize']=="LE3"]=1

df['Pstatus'][df['Pstatus']=="A"]=0
df['Pstatus'][df['Pstatus']=="T"]=1

df['Mjob'][df['Mjob']=="teacher"]=1
df['Mjob'][df['Mjob']=="health"]=2
df['Mjob'][df['Mjob']=="services"]=3
df['Mjob'][df['Mjob']=="at_home"]=4
df['Mjob'][df['Mjob']=="other"]=5

df['Fjob'][df['Fjob']=="teacher"]=1
df['Fjob'][df['Fjob']=="health"]=2
df['Fjob'][df['Fjob']=="services"]=3
df['Fjob'][df['Fjob']=="at_home"]=4
df['Fjob'][df['Fjob']=="other"]=5

df['reason'][df['reason']=="home"]=1
df['reason'][df['reason']=="reputation"]=2
df['reason'][df['reason']=="course"]=3
df['reason'][df['reason']=="other"]=4

df['guardian'][df['guardian']=="mother"]=1
df['guardian'][df['guardian']=="father"]=2
df['guardian'][df['guardian']=="other"]=3

df['guardian'][df['guardian']=="father"]=2
df['guardian'][df['guardian']=="other"]=3

for i in df:
    df[i][df[i]=="no"]=0
    df[i][df[i]=="yes"]=1

X = df.drop('G3',axis=1).values
y = df['G3'].values

# print(df.shape)

# train test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=42)


# Setup arrays to store training and test accuracies
neighbors = np.arange(1,33)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


for i,k in enumerate(neighbors):
    # Setup a knn classifier with k neighbors
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit the model
    knn.fit(X_train, y_train)
    
    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    # Compute accuracy on the test set
    test_accuracy[i] = knn.score(X_test, y_test) 


# Generate plot
plt.title('k-NN Varying number of neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.show()

# Get maximum testing accuracy for k=11
# Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=11)

# Fit the model
print("Fit: ", knn.fit(X_train,y_train))

#Get accuracy. Note: In case of classification algorithms score method represents accuracy.
print("Score: ", knn.score(X_test,y_test))

# let us get the predictions using the classifier we had fit above
y_pred = knn.predict(X_test)
# confusion_matrix(y_test,y_pred)
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


# =========== Classification Report ===========

print(classification_report(y_test,y_pred))


# =========== GridSearchCV ===========

# In case of classifier like knn the parameter to be tuned is n_neighbors
param_grid = {'n_neighbors':np.arange(1,33)}

knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
print("GridSearchCV fit: ")
print(knn_cv.fit(X,y))

print("GridSearchCV best score: ")
print(knn_cv.best_score_)

print("GridSearchCV best params: ")
print(knn_cv.best_params_)




# =========== Logistic Regression ===========

print("============ LR ============")

lr = LogisticRegression()
scaler = StandardScaler()
model1 = Pipeline([('standardize', scaler),
                    ('log_reg', lr)])
model1.fit(X_train, y_train)

y_train_hat = model1.predict(X_train)
y_train_hat_probs = model1.predict_proba(X_train)[:,1]
# confusion_matrix(y_train, y_train_hat)

y_test_hat = model1.predict(X_test)
y_test_hat_probs = model1.predict_proba(X_test)[:,1]
# confusion_matrix(y_test, y_test_hat)
print(pd.crosstab(y_test, y_test_hat, rownames=['True'], colnames=['Predicted'], margins=True))


# =========== Classification Report ===========

print(classification_report(y_test, y_test_hat))
