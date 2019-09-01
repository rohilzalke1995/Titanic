# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:50:28 2019

@author: rohil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('Titanic.csv')

df = df.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'PassengerId'], axis = 'columns')


df['Age'] = df['Age'].fillna(df['Age'].mean())


x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
Le = LabelEncoder()
x[:, 1] = Le.fit_transform(x[:, 1])

Ohe = OneHotEncoder(categorical_features=[1])
x = Ohe.fit_transform(x).toarray()


from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

x = x[:, 1:]

from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
Knn.fit(x_train, y_train)

y_pred = Knn.predict(x_train)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_train, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_pred)

c = df.corr()
c['Survived'].sort_values(ascending = False)




Knn.predict([[0, 1, 1, 40, 100]])