# Data Preprocessing

Update 18/3/19: Categorical variables: Replaced all columns of type 'object' with 'categorical'

Printed top 10 most frequent values for each categorical feature to look at distribution; identified features that may be worthwhile investigating due to skewed distribution
                                       
Created a threshold function that removes columns below a certain proportion of non-null values; even if we keep columns with at least 5% of non-null values then we remove 135 out of 212 columns, so this is probably the best way to do dimensionality reduction

# Preprocess numerical features & generate graphs

Run project_preprocessed_visualised.ipynb in jupyter notebook to preprocess numerical features and generate graphs in report.

Files in this folder are the work by S1890666 and S1887468.
