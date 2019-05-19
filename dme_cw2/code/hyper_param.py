# -*- coding: utf-8 -*-
'''
Hyper-parameter selection

Author:
organize all code: S1802373, S1809576
'''
import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import itertools
import collections
import matplotlib.ticker as mtick
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def nz(value):

    '''
    Convert None to int zero else return value.
    '''

    if value == None:
        return 0
    return value

small_raw = pd.read_csv('../data/feature/orange_small_train.data', sep = '\t')

    #Check and remove empty columns
small=small_raw.dropna(how='all', axis=1) #drop empty columns
removed_columns = small_raw.shape[1] - small.shape[1]
#print("Removed empty columns: " + str(removed_columns)) #confirm how many columns were removed

#Check if there are duplicate columns in the remaining dataset:
dup_count =small.columns.duplicated() #array of true/false for duplicated/non-duplicated columns
d=collections.Counter(dup_count)
#print("Number of duplicate columns:"+str(nz(d.get(True)))) #use none to zero function

#print(small.info())

#Show dataset statistics again:
pd.set_option('display.max_columns', 230)
#display(small.describe()) #shows the dataset statistics
#display(small.head(5))

#Check the number of unique entries per column:
pd.set_option('display.max_rows', 230)
#small.nunique()

#Replacing object columns with categorical

small_copy = small.copy(deep=True)
for var in small_copy.columns:
    if small_copy[var].dtype == 'object':
        cat_col = small_copy[var].astype('category')
        small_copy.loc[:,var] = cat_col
        # print('Categories for ',var, ': ',small_copy[var].cat.categories)

#ax = sns.countplot(x='Var197', data=small_copy)
cat_indexes = []
for i in range(191,230,1):
    cat_indexes.append('Var'+str(i))

cat_indexes.remove('Var209')

def removenull(data, threshold):
    datacopy = data.copy(deep=True)
    datacopy2 = datacopy.dropna(axis=1, thresh=int((datacopy.shape[0])*threshold))
    # print('Removed ', data.shape[1]-datacopy2.shape[1], 'columns using threshold ', '{0:.2f}'.format(threshold))
    return datacopy2

#print(removenull(small_copy, 0.9).describe())
#print(removenull(small_copy, 0.5).describe())
#print(removenull(small_copy, 0.2).describe())

# print('Original number of columns: ', small_copy.shape[1])
for i in np.arange(0,1,0.05):
    removenull(small_copy, i)

#Loads in all three targets into a separate dataframe
app_labels = pd.read_csv('../data/label/orange_small_train_appetency.csv', header=None)
churn_labels = pd.read_csv('../data/label/orange_small_train_churn.csv', header=None)
upsell_labels = pd.read_csv('../data/label/orange_small_train_upselling.csv', header=None)
small_labels = pd.concat([app_labels, churn_labels, upsell_labels], axis=1)
# print(small_labels.head(10))
small_labels.columns = ['Appetency', 'Churn','Upselling']
# print(small_labels.head(10))

small_1= small_copy.loc[:, small_copy.dtypes == np.float64]
for col in small_1.columns:
    small_1[col].fillna((small_1[col].mean()), inplace=True)

#create correlation matrix
corr_matrix = small_1.loc[:, small_1.dtypes == np.float64].corr().abs()

#Set a maximum cap for correlation
corr_cap = 0.8 #max acceptable correlation

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than max acceptable
to_drop1 = [column for column in upper.columns if any(upper[column] > corr_cap)]  #float columns

# print('Total highgly correlated columns to remove:'+str(len(to_drop1)))
# print(to_drop1)

#Drop correlated columns, show df head and the matrix

small_2=small_1.drop(small_1[to_drop1], axis=1)
c=(small_2.loc[:, small_1.dtypes == np.float64]).shape

# print('Float columns kept:'+str(c[1]))

#Visualise the new correlation matrix
corr_matrix2 = small_2.loc[:, small_1.dtypes == np.float64].corr().abs()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

categorical_columns = small_copy.select_dtypes(include=['category'])
#print(categorical_columns.info())
small_3 = pd.concat([small_2,categorical_columns], axis=1)

def replaceMissing(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    for feature in category_features:
        missing_values = df[feature].isnull()
        if missing_values.sum() > 0:
            df[feature].cat.add_categories('MISSING', inplace=True)
    return

def collapseCategories(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    collpased_features = {}
    for feature in category_features:
        col = df[feature].copy()
        for category in col.cat.categories:
            category_index = col == category
            if category_index.sum() < 0.05 * len(col):
                if feature not in collpased_features:
                    collpased_features[feature] = []
                collpased_features[feature].append(category)

                if 'OTHERS' not in df[feature].cat.categories:
                    df[feature].cat.add_categories('OTHERS', inplace=True)
                df.loc[category_index, feature] = 'OTHERS'
                df[feature].cat.remove_categories(category, inplace=True)
    return collpased_features

def getGoodCategories(df, category_features):
    if category_features is None:
        category_features = [colname for colname in df.columns if df[colname].dtype != float]
    feature_to_remove = set()
    for feature in category_features:
        categories = df[feature].cat.categories
        if len(categories) == 1 or len(set(categories) - set(['MISSING', 'OTHERS'])) < 2:
            feature_to_remove.add(feature)
    return list(set(category_features) - feature_to_remove)

def collapseGivenCategories(df, collapse_features):
    for feature in collapse_features:
        if 'OTHERS' not in df[feature].cat.categories:
            df[feature].cat.add_categories('OTHERS', inplace=True)

        col = df[feature].copy()
        for category in collapse_features:
            category_index = col == category
            if category_index.sum() != 0:
                df.loc[category_index, feature] = 'OTHERS'
                df[feature].cat.remove_categories(category, inplace=True)
    return

category_features = [colname for colname in small_copy.columns if small_copy[colname].dtype != float]

# Check how many categories each categorical variable has
category_features_levels = small_copy[category_features[1:]].apply(lambda col: len(col.cat.categories))
# print(category_features_levels)

# Some categorical variables are with too many categories, we can exclude those features out from the dataset
category_features = category_features_levels[category_features_levels <= 500].index
# print(category_features)

replaceMissing(small_copy, category_features)
collapse_features = collapseCategories(small_copy, category_features)

category_features = getGoodCategories(small_copy, category_features)

small_4 = pd.concat([small_2, small_copy[category_features]], axis=1)

X_dummy = pd.get_dummies(small_4)

pca = PCA(n_components=32)
X = pca.fit_transform(X_dummy)

app_label_path = "../data/label/orange_small_train_appetency.labels"
chu_label_path = "../data/label/orange_small_train_churn.labels"
ups_label_path = "../data/label/orange_small_train_upselling.labels"
y_app = pd.read_csv(app_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_app.cat.rename_categories(['-1', '1'], inplace=True)
y_chu = pd.read_csv(chu_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_chu.cat.rename_categories(['-1', '1'], inplace=True)
y_ups = pd.read_csv(ups_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_ups.cat.rename_categories(['-1', '1'], inplace=True)

#GradientBoostingClassifier
# gbrt_param = [{'n_estimators': [10, 20, 30, 50], 'max_depth': [5, 8, 10, 15], 'learning_rate': [0.05, 0.1, 0.2, 0.5]}]
# gbrt = GridSearchCV(GradientBoostingClassifier(), gbrt_param, cv = 5)
# gbrt.fit(X, y_app)
# print(gbrt.best_params_)

# scores = [x[1] for x in rfc.grid_scores_]
# scores = np.array(scores).reshape(len(esti), len(depth))
#
# for ind, i in enumerate(esti):
#     plt.plot(depth, scores[ind], label='esti: ' + str(i))
# plt.legend()
# plt.xlabel('depth')
# plt.ylabel('Mean score')
# plt.show()

# esti = [10, 20, 30, 50, 100, 200]
# depth = [5, 8, 10, 15]
# rfc = GridSearchCV(RandomForestClassifier(), dict(n_estimators = esti, max_depth = depth), cv = 5, n_jobs = 2)
# rfc.fit(X, y_app)

# test
print("start selection")
parameter_grid_dt = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

parameter_grid_lr = {'tol': [0.1, 1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000],
                     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                     'max_iter': [100, 1000, 10000]}

parameter_grid_rfc = {'n_estimators': [10, 20, 30, 50, 100, 200],
                      'max_depth': [5, 8, 10, 15],
                      'max_features': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}

parameter_grid_AdaDT = {'n_estimators': [10, 20, 30, 50],
                        'learning_rate': [0.05, 0.1, 0.2, 0.5]}

grid_search_dt = GridSearchCV(DecisionTreeClassifier(), parameter_grid_dt, cv = 5, n_jobs=-1)
grid_search_lr = GridSearchCV(LogisticRegression(), parameter_grid_lr, cv = 5, n_jobs=-1)
grid_search_rfc = GridSearchCV(RandomForestClassifier(), parameter_grid_rfc, cv = 5, n_jobs=-1)
grid_search_AdaDT = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=0, max_depth=8),random_state=0), parameter_grid_AdaDT, cv = 5, n_jobs=-1)

grid_file_path = "../result/text/hyper_param.txt"
if os.path.isfile(grid_file_path):
    os.remove(grid_file_path)

print("dt")
grid_search_dt.fit(X, y_app)
with open(grid_file_path, "a") as f:
    f.write("Decision Tree" + "\n")
    f.write("Best Score: {} \n".format(grid_search_dt.best_score_))
    f.write("Best params: {} \n".format(grid_search_dt.best_params_))
    # f.write("cv_results_: \n")
    # for key, value in grid_search_dt.cv_results_:
    #     f.write(key + ": " + value + "\n")

# print("lr")
# grid_search_lr.fit(X, y_app)
# with open(grid_file_path, "a") as f:
#     f.write("Logistic Regression" + "\n")
#     f.write("Best Score: {} \n".format(grid_search_lr.best_score_))
#     f.write("Best params: {} \n".format(grid_search_lr.best_params_))
    # f.write("cv_results_: \n")
    # for key, value in grid_search_lr.cv_results_:
    #     f.write(key + ": " + value + "\n")

print("rfc")
grid_search_rfc.fit(X, y_app)
with open(grid_file_path, "a") as f:
    f.write("Random Forest Classifier" + "\n")
    f.write("Best Score: {} \n".format(grid_search_rfc.best_score_))
    f.write("Best params: {} \n".format(grid_search_rfc.best_params_))
    # f.write("cv_results_: \n")
    # for key, value in grid_search_rfc.cv_results_:
    #     f.write(key + ": " + value + "\n")

print("AdaDT")
grid_search_AdaDT.fit(X, y_app)
with open(grid_file_path, "a") as f:
    f.write("AdaBoost Classifier (DecisionTreeClassifier)" + "\n")
    f.write("Best Score: {} \n".format(grid_search_AdaDT.best_score_))
    f.write("Best params: {} \n".format(grid_search_AdaDT.best_params_))
    # f.write("cv_results_: \n")
    # for key, value in grid_search_AdaDT.cv_results_:
    #     f.write(key + ": " + value + "\n")

# print(grid_search.cv_results_["params"])
# print(grid_search.cv_results_["mean_test_score"])
# print(grid_search.cv_results_["std_test_score"])
# print("Best Score: {}".format(grid_search.best_score_))
# print("Best params: {}".format(grid_search.best_params_))
