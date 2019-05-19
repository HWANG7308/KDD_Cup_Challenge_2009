# How to Run The Code

1.cd code/

2.python preprocessing.py

3.cd ../scripts/

4.bash test.sh

There will be more detailed guide on usage.

After running, the graph results with be in "result/graph", some text results will be in "result/text".

# How we contribute to our code

## Preprocessing data on numerical features
rushingdragging and S1887468 provide files in folder './data_preprocessing_temp'

## Preprocessing data on categorical features & PCA
philipppp000 provides code in 'preprocessing.py'

## Classification method & integration of all code files
HWANG7308 provides code for classification model and organizes all our code with other necessary part to make our code running better

# Preliminary
## MARKETING: PREDICTING CUSTOMER CHURN
### Dataset

Dataset here: 

http://www.vincentlemaire-labs.fr/kddcup2009/#data

### Task 

This data was used in the KDD Cup 2009. The task is to predict customer behaviour, namely, churn (whether they will switch providers), appetancy (whether they will buy new products), and whether they may be willing to buy upgrades.

### Size

Number of data items: 50,000. Number of features: 15,000 (large data set), 230 (small data set). For this task we recommend that you start with the small data.

### Note

You may wish to focus on only one of the three prediction tasks (churn, appetancy, or upselling). Alternatively, if you are ambitious, you may try predicting all three in one classifier.

### Challenges

Large number of features, data may be noisy, due to anonymity constraints you do not have a lot of information about what the features mean.

# References

1.New website for the KDD-cup

https://www.kdd.org/kdd-cup/view/kdd-cup-2009/Data

2.Datasets here

http://www.vincentlemaire-labs.fr/kddcup2009/#data

3.Proceedings of KDD-Cup 2009 competition

http://proceedings.mlr.press/v7/

4.KDD-Cup 2009 competition web site (Note post-challenge entries still ranked on the leaderboard!)

https://www.hugedomains.com/domain_profile.cfm?d=kddcup-orange&e=com

5.Kaggle competition

https://www.kaggle.com/asminalev/kdd-cup-2009-customer-relationship-prediction/kernels
