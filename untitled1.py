# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:39:19 2019

@author: Yanyan
"""

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
people = fetch_lfw_people(min_faces_per_person=20,resize=0.7) 
X_people = people.data
y_people = people.target
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, 
                                            stratify=y_people, random_state=0) 
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train,y_train)
print("Test set score of xxx: {:.2f}".format(lda.score(X_test, y_test)))