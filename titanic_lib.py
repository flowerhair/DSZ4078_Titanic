# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:09:36 2025

@author: marti
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score
import pandas as pd

def predict_cat_feature(feature, predictors):
    getX = predictors[pd.isna(feature) == False]
    getY = feature[pd.isna(feature) == False]
    
    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(getX, getY, test_size=0.2, random_state=42, stratify=getY)
    
    #train model - decision tree classifier
    tree=DecisionTreeClassifier()
    tree_para = {'criterion':['gini','entropy'],'max_depth':[5,7,9,10,20], 'min_samples_leaf':[2,3,5]}
    clf = GridSearchCV(tree, tree_para, scoring='f1_samples', cv=5)
    clf.fit(X_train, y_train)
    best_tree = clf.best_estimator_
    return (best_tree, f1_score(y_test, best_tree.predict(X_test)))
    

def predict_cont_feature(feature, predictors):  
    getX = predictors[feature.isna() == False]
    getY = [feature.isna() == False]
    
    #split the dataset
    X_train, X_test, y_train, y_test = train_test_split(getX, getY, test_size=0.2, random_state=42, stratify=getY)
    
    #train model - decision tree classifier
    tree=DecisionTreeRegressor()
    tree_para = {'criterion':['gini','entropy'],'max_depth':[5,7,9,10,20], 'min_samples_leaf':[2,3,5]}
    clf = GridSearchCV(tree, tree_para, scoring='r2', cv=5)
    clf.fit(X_train, y_train)
    best_tree = clf.best_estimator_
    return (best_tree, r2_score(y_test, best_tree.predict(X_test)))
    