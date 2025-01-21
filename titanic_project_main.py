# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 12:40:32 2025

@author: marti
"""

#importy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import f1_score, r2_score, classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack

from titanic_lib import predict_cat_feature, predict_cont_feature 

df_train = pd.read_csv('./files/titanic_train.csv')
df_test = pd.read_csv('./files/titanic_test.csv')

#exploratory analysis
print(df_train.info())
df_train.describe()
df_train.duplicated().sum()
pd.set_option('display.max_columns', None)

#handy table
basic_info=pd.DataFrame({
     "NaNs": df_train.isnull().sum(),
     "Uniques": df_train.nunique(),
     "Counts": df_train.count(),
     "Types": df_train.dtypes
     })

basic_info_test=pd.DataFrame({
     "NaNs": df_test.isnull().sum(),
     "Uniques": df_test.nunique(),
     "Counts": df_test.count(),
     "Types": df_test.dtypes
     })
#chybí dvě hodnoty v embarked - doplním
#chybí 177 hodnot v age - porovnám když doplním a když vynechám age. viz grafy versus age

#nejdřív ale omezím data - cabin vynechám, spousta chybějících hodnot
#ticket - vynechám, stálo by spoustu času to nějak rozklíčovat, spoustu hodnot, různý formát atd
#fare uvidím podle korelace

#featur engineering
#z name zkusím vyextrahovat title a porovnat jestli je n+jaká korelace
#nejdřív ale připravím spojený ddataset train i test abych dopočítal age a embarked
df_combined = pd.concat([df_train, df_test], ignore_index=True)

#doplním jedinnou chybějící hodnotu pro fare jako median pro danou tkkřídu
fare_median = df_combined[["Pclass", "Fare"]].groupby("Pclass").median().to_dict(orient='dict', index=True)['Fare']
df_combined["Fare_c"]=df_combined.apply(
    lambda x: fare_median[x["Pclass"]] if pd.isna(x["Fare"]) else x["Fare"], axis=1
    )

df_combined[pd.isna(df_combined["Embarked"])==False].shape

df_combined["Title"]=df_combined.apply(lambda x: x["Name"].split(', ')[1].split('.')[0], axis=1)


plt.figure()
sns.kdeplot(x=df_combined[df_combined["Age"].notna()]["Age"], hue=df_combined[df_combined["Age"].notna()]["Title"])
plt.title("Age versus title")
plt.show()

#master je zjevně mladý kluk, může se hodit pro doplění age. pokud vůbecx bude nějaká korelace age v survived
map_condensed_title = {
    'Mr': 1,
	'Miss': 4,
	'Mrs': 1,
	'Master': 5,
	'Rev': 2,
	'Dr': 2,
	'Col': 2,
	'Ms': 4,
	'Major': 2,
	'Mlle': 4,
	'Sir': 3,
	'Capt': 2,
	'Mme': 1,
	'Lady': 3,
	'Jonkheer': 2,
	'Dona': 3,
	'Don': 3,
	'the Countess': 3,
    }

df_combined[df_combined["Title"]=="Mlle"]

df_combined["CondTitle"]=df_combined["Title"].apply(lambda x: map_condensed_title[x])

con_cols = ["Age", "Fare_c"]
cat_cols = ["SibSp", "Parch", "Sex", "Pclass", "CondTitle", "Embarked"]

for feature in con_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(x=df_combined[feature], hue=df_combined["Survived"])
    plt.title(f'{feature} vs Target')
    plt.show()

for feature in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_combined[feature], hue=df_combined["Survived"])
    plt.title(f'{feature} vs Target')
    plt.show()

cat_cols2 = ["SibSp", "Parch", "Sex", "Pclass", "CondTitle"]

for feature in cat_cols2:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df_combined[feature], hue=df_combined["Embarked"])
    plt.title(f'{feature} vs Target')
    plt.show()

for feature in con_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(x=df_combined[feature], hue=df_combined["Embarked"])
    plt.title(f'{feature} vs Target')
    plt.show()    

plt.figure(figsize=(6, 4))
sns.countplot(x=df_combined["CondTitle"], hue=df_combined["Pclass"])
plt.title('Title vs Class')
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(x=df_combined["Age"], hue=df_combined["Embarked"])
plt.title('Age vs Embarked')
plt.show()

plt.figure(figsize=(6, 4))
sns.kdeplot(x=df_combined["Fare"], hue=df_combined["Title"])
plt.title('Fare vs Class')
plt.show()

plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_combined["Fare"], y=df_combined["Age"], hue=df_combined["Pclass"])
plt.title('Fare vs Age')
plt.show()

#chci vidět jestli max fare záznam dává smysl - dává , podle všeho není outlier
#df_combined[df_combined["Fare"]==df_combined["Fare"].max()]
    
#train model - decision tree classifier
#sklearn decision tree requires numerical data, so i have to do some preparation

#data prep

#convert categorical fields to numerical values
map_gender = {'male': 1, 'female': 2}


df_combined["Sex_c"]=df_combined["Sex"].apply(lambda x: map_gender[x])

df_conv = df_combined.drop(columns=["Title", "Ticket", "Cabin", "Name", "Fare", "PassengerId", "Sex"])


#teď potřebuju doplnit hodnoty embarked a age a potom udělat split a modelování dat

#-----------------Embarked doplnění------------------------
#ppůvodně jsem zkoušel onehotencoding ale výsledek byl mnohme horší. 
#pro randomforest and decisiontree, labvelencoding je lepší - menší cardinalita


features_for_embarked = ["Fare_c", "SibSp", "Parch", "Pclass", "CondTitle"]

getX = df_conv[df_conv["Embarked"].notna()][features_for_embarked]
getY = df_conv[df_conv["Embarked"].notna()]["Embarked"]
    
#split the dataset
X_train, X_test, y_train, y_test = train_test_split(getX, getY, test_size=0.2, random_state=42, stratify=getY)


#náhodný les
clf = RandomForestClassifier(random_state=42)

#optimalizace hyperparametrů
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [2, 4, 6]
}

# Grid search
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Use all available cores
)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

best_embarked_model = grid_search.best_estimator_

y_pred = best_embarked_model.predict(X_test)

cm_embarked = confusion_matrix(y_test, y_pred, labels=best_embarked_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_embarked, display_labels=best_embarked_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

y_train_pred = best_embarked_model.predict(X_train)
print(confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))
#nějak se musím rozhodnot kdy použiju ten  model a kdy bych vynechal data nebo nějak doplnil ručně
#konkrétně tady mi ten model nevychází nijak extra dobře, ale nechci už nad tím trávit tolik času. 
#kdybych to doplňoval od oka, tak bych tam dal stejné hodnoty jako to vyšlo...

#print těch řádků kde chybí embarked před doplněním
df_conv[df_conv["Embarked"].isna()]

#náhrada c hybějících hodnot v původním sloupci
df_emb_replaced = df_conv[df_conv["Embarked"].isna()]
df_emb_replaced["Embarked"]=best_embarked_model.predict(df_emb_replaced[features_for_embarked])
#tohle nahrazení dává varování, mělo by se dělat pomocí .loc
#df_emb_replaced.loc[:, "Embarked"]=best_embarked_model.predict(df_emb_replaced[["Pclass", "SibSp", "Parch", "Fare_c", "CondTitle"]])

#print po náhradě chybějících hodnot
df_emb_replaced

#-------------------------Age doplnění - už mám přidaný embarked z předchozího kroku---------

#data kde embarked byl a spojím s dopočítanými daty
df_emb=pd.concat([df_emb_replaced, df_conv[df_conv["Embarked"].notna()]])
df_emb[df_emb["Embarked"].isna()].shape
df_emb.shape

#potřebuju sloupec Embarked převést na numerické hodnoty
#po zkoušení a hraní si s encoding je lepší jednoduše převést pomocí slovníku
map_embarked = {'C': 1, 'Q': 2, 'S': 3}

#udělám si obrázek na kterých sloupcích závisí Age
cols = ["Sex_c", "CondTitle", "Pclass", "SibSp", "Parch", "Embarked"]

for feature in cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(x=df_emb[df_emb["Age"].notna()]["Age"], hue=df_emb[df_emb["Age"].notna()][feature])
    plt.title(f'{feature} vs Age')
    plt.show()
    
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df_emb[df_emb["Age"].notna()]["Fare_c"], y=df_emb[df_emb["Age"].notna()]["Age"])
plt.title('Fare vs Age')
plt.show()


df_emb.loc[:,"Embarked"] = df_emb["Embarked"].apply(lambda x: map_embarked[x])
df_emb['Embarked'] = pd.to_numeric(df_emb['Embarked'], downcast='integer', errors='coerce')
df_emb.info()

grouped = df_emb[["Pclass","CondTitle", "Age"]].groupby(["Pclass", "CondTitle"]).median()
group_reset = grouped.reset_index()
group_reset.rename(columns={"Age": "NewAge"}, inplace=True)
df_emb = df_emb.merge(group_reset, on=["Pclass", "CondTitle"], how='left')
df_emb["Age_c"] = df_emb.apply(lambda x: x["NewAge"] if pd.isna(x["Age"]) else x["Age"], axis=1)

#finální dataframe , numerické hodnoty ve všech polích, doplněné chybějící hodnoty
df = df_emb.drop(columns=["Age", "NewAge"])

"""

plt.figure()
pd.qcut(df_emb_conv['Age'], q=5).value_counts().plot(kind='bar', title='Equal Frequency')
plt.show()

binned_age = pd.qcut(df_emb_conv['Age'], q=5, labels=False)
tohle nějak moc nefunguje
vyzkouším age nahradit nějakým průměrem a porovnat model s age a bez age sloupce
X_age = df_emb_conv[df_emb_conv["Age"].notna()][["CondTitle", "Pclass", "Embarked"]]
y_age = df_emb_conv[df_emb_conv["Age"].notna()]["Age_b"]


#split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_age, y_age, test_size=0.2, random_state=42, stratify=y_age)


#náhodný les
clf_age = RandomForestClassifier(random_state=42)

#optimalizace hyperparametrů
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 10],
    'min_samples_split': [3, 5, 7],
    'min_samples_leaf': [2, 4, 6],
    'max_features': ['sqrt', 3]
}

# Grid search
grid_search_age = GridSearchCV(
    estimator=clf_age,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=2,
    n_jobs=-1  # Use all available cores
)



grid_search_age.fit(X_train, y_train)

best_age_model = grid_search_age.best_estimator_

y_pred_age = best_age_model.predict(X_test)

print(classification_report(y_test, y_pred_age))
print(confusion_matrix(y_test, y_pred_age))
"""

#--------------------------------------- Finální příprava pro trénování-------------------
#Rozdělení dat na trénovací, testovací a to se co má určit
#asi se použije randomforest, nemám data připravená pro KNN 
df.shape
X = df[df["Survived"].notna()]
X = X.drop(columns= ["Survived"])
X.shape
y = df[df["Survived"].notna()]["Survived"]
y.shape
X_guess = df[df["Survived"].isna()]
X_guess.shape

plt.figure(figsize=(6, 4))
sns.kdeplot(x=df["Fare_c"], hue=df["Survived"])
plt.title('Fare vs Survived')
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X_age, y_age, test_size=0.2, random_state=42, stratify=y_age)