# -*- coding: utf-8 -*-
'''
Preprocessing on original data

Author:
preprocessing on numerical feature: S1890666, S1887468
preprocessing on categorical feature: S1809576
organize all code: S1802373
'''

import sys
import os
import sys
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# preprocessing on numerical feature
#Functions
def nz(value):

    '''
    Convert None to int zero else return value.
    '''

    if value == None:
        return 0
    return value

#Raw data initial exploration
#Load small dataset
train_data_path = "../data/feature/orange_small_train.data"
small_raw = pd.read_csv(train_data_path, sep = '\t')

#Check and remove empty columns
small=small_raw.dropna(how='all', axis=1) #drop empty columns
removed_columns = small_raw.shape[1] - small.shape[1]

#Check if there are duplicate columns in the remaining dataset:
dup_count =small.columns.duplicated() #array of true/false for duplicated/non-duplicated columns
d=collections.Counter(dup_count)

#Show dataset statistics again:
# pd.set_option('display.max_columns', 230)

#Check the number of unique entries per column:
# pd.set_option('display.max_rows', 230)

#Replacing object columns with categorical
small_copy = small.copy(deep=True)
for var in small_copy.columns:
    if small_copy[var].dtype == 'object':
        cat_col = small_copy[var].astype('category')
        small_copy.loc[:,var] = cat_col
        print('Categories for ',var, ': ',small_copy[var].cat.categories)

cat_indexes = []
for i in range(191,230,1):
    cat_indexes.append('Var'+str(i))

cat_indexes.remove('Var209')

# Remove columns with a certain threshold of null values
#Threshold should be a proportion of how many values needed to keep column
#i.e. if we only want columns with 70% non-null values or more, threshold should be 0.7
def removenull(data, threshold):
    datacopy = data.copy(deep=True)
    datacopy2 = datacopy.dropna(axis=1, thresh=int((datacopy.shape[0])*threshold))
    print('Removed ', data.shape[1]-datacopy2.shape[1], 'columns using threshold ', '{0:.2f}'.format(threshold))
    return datacopy2

print('Original number of columns: ', small_copy.shape[1])
for i in np.arange(0,1,0.05):
    removenull(small_copy, i)

# Select Float columns, create indicators of missing values, replace NaN with mean
# output to small_f
small_1= small_copy.loc[:, small_copy.dtypes == np.float64]
for col in small_1.columns:
    # small_1[col+"_missing"] = small_1[col].isnull().astype(int) # comment out this line and there will be no missing matrix
    small_1[col].fillna((small_1[col].mean()), inplace=True)

#create correlation matrix
corr_matrix = small_1.loc[:, small_1.dtypes == np.float64].corr().abs()

#Set a maximum cap for correlation
corr_cap = 0.8 #max acceptable correlation

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find index of feature columns with correlation greater than max acceptable
to_drop1 = [column for column in upper.columns if any(upper[column] > corr_cap)]  #float columns

#Drop correlated columns, show df head and the matrix

small_2=small_1.drop(small_1[to_drop1], axis=1)
c=(small_2.loc[:, small_1.dtypes == np.float64]).shape

#Visualise the new correlation matrix
corr_matrix2 = small_2.loc[:, small_1.dtypes == np.float64].corr().abs()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix2, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


categorical_columns = small_copy.select_dtypes(include=['category'])
small_3 = pd.concat([small_2,categorical_columns], axis=1)
# the above codes are the work by S1890666, S1887468

# preprocessing on categorical features by S1809576 below
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

# Some categorical variables are with too many categories, we can exclude those features out from the dataset
category_features = category_features_levels[category_features_levels <= 500].index

replaceMissing(small_copy, category_features)
collapse_features = collapseCategories(small_copy, category_features)

category_features = getGoodCategories(small_copy, category_features)

app_label_path = "../data/label/orange_small_train_appetency.labels"
chu_label_path = "../data/label/orange_small_train_churn.labels"
ups_label_path = "../data/label/orange_small_train_upselling.labels"
y_app = pd.read_csv(app_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_app.cat.rename_categories(['-1', '1'], inplace=True)
y_chu = pd.read_csv(chu_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_chu.cat.rename_categories(['-1', '1'], inplace=True)
y_ups = pd.read_csv(ups_label_path, sep = '\t', header=None).iloc[:, 0].astype('category')
y_ups.cat.rename_categories(['-1', '1'], inplace=True)

# pca = PCA(n_components=32) # ---------------------------------

# ---------- feature (no 0/1; PCA on all features) ----------
# small_4_n_pca_all = pd.concat([small_2, small_copy[category_features]], axis=1)
# X_dummy_n_pca_all = pd.get_dummies(small_4_n_pca_all)
# # X_dummy_n_pca_all = pca.fit_transform(X_dummy_n_pca_all)
# X_dummy_n_pca_all = pd.DataFrame(X_dummy_n_pca_all)
#
# # pca = PCA().fit(X_dummy_n_pca_all)
# # plt.clf()
# # plt.plot(np.cumsum(pca.explained_variance_ratio_))
# # plt.xlabel('number of components')
# # plt.ylabel('cumulative explained variance')
# # plt.xlim((-1,40))
# # plt.show()
#
# X_y_chu_n_pca_all = pd.concat([X_dummy_n_pca_all, y_chu], axis = 1)
# X_y_app_n_pca_all = pd.concat([X_dummy_n_pca_all, y_app], axis = 1)
# X_y_ups_n_pca_all = pd.concat([X_dummy_n_pca_all, y_ups], axis = 1)
#
# app_path_n_pca_all = "../data/feature/small_train_appetency_n_pca_all.csv"
# chu_path_n_pca_all = "../data/feature/small_train_churn_n_pca_all.csv"
# ups_path_n_pca_all = "../data/feature/small_train_upselling_n_pca_all.csv"
#
# X_y_app_y_pca_all.to_csv(app_path_n_pca_all, header=None, index=None)
# print("appetency prepared n_pca_all")
# X_y_chu_y_pca_all.to_csv(chu_path_n_pca_all, header=None, index=None)
# print("churn prepared n_pca_all")
# X_y_ups_y_pca_all.to_csv(ups_path_n_pca_all, header=None, index=None)
# print("upselling prepared n_pca_all")

# ---------- feature (no 0/1; PCA on catrgorical features) ----------
# one-hot on categorical features
cat_dummy_y_pca_cat = pd.get_dummies(small_copy[category_features])
# PCA on one-hoted categorical features
# cat_dummy_y_pca_cat = pca.fit_transform(cat_dummy_y_pca_cat) # ---------------------------------
cat_dummy_y_pca_cat = pd.DataFrame(cat_dummy_y_pca_cat)
small_4_y_pca_cat = pd.concat([small_2, cat_dummy_y_pca_cat], axis=1)

print(small_copy[category_features])
print(cat_dummy_y_pca_cat)
print(small_2)

# pca = PCA().fit(cat_dummy_y_pca_cat)
# plt.clf()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.xlim((-1,40))
# plt.show()

# X_y_chu_y_pca_cat = pd.concat([small_4_y_pca_cat, y_chu], axis = 1)
# X_y_app_y_pca_cat = pd.concat([small_4_y_pca_cat, y_app], axis = 1)
# X_y_ups_y_pca_cat = pd.concat([small_4_y_pca_cat, y_ups], axis = 1)
#
# app_path_y_pca_cat = "../data/feature/small_train_appetency_n_no_pca.csv"
# chu_path_y_pca_cat = "../data/feature/small_train_churn_n_no_pca.csv"
# ups_path_y_pca_cat = "../data/feature/small_train_upselling_n_no_pca.csv"
#
# X_y_app_y_pca_cat.to_csv(app_path_y_pca_cat, header=None, index=None)
# print("appetency prepared y_pca_cat")
# X_y_chu_y_pca_cat.to_csv(chu_path_y_pca_cat, header=None, index=None)
# print("churn prepared y_pca_cat")
# X_y_ups_y_pca_cat.to_csv(ups_path_y_pca_cat, header=None, index=None)
# print("upselling prepared y_pca_cat")
