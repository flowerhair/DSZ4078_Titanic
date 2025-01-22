# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 14:09:36 2025

@author: marti
"""
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, r2_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
import pandas as pd

def test_rfc(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rfc_surv = RandomForestClassifier(random_state=42)
    
    #optimalizace hyperparametrů
    param_grid_surv = {
        'n_estimators': [100, 300, 500, 1000],
        'max_depth': [5, 7, 10,15],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 3, 5]
    }
    
    # Grid search
    grid_search_surv = GridSearchCV(
        estimator=rfc_surv,
        param_grid=param_grid_surv,
        cv=5,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1  # Use all available cores
    )
    
    grid_search_surv.fit(X_train, y_train)
    
    best_surv_model = grid_search_surv.best_estimator_
    best_params = grid_search_surv.best_params_
    y_pred_surv = best_surv_model.predict(X_test)
    cr = classification_report(y_test, y_pred_surv)
    cm = confusion_matrix(y_test, y_pred_surv)
    
    return_dict = {'estimator': best_surv_model,
                   'paramas': best_params,
                   'cr': cr,
                   'cm': cm
                   }
    return return_dict
    