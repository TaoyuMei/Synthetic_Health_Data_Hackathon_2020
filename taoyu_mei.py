# define imports
import pandas as pd
import numpy as np

import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# from sklearn.preprocessing import OneHotEncoder

# set working directory
os.getcwd()
os.chdir("Diabetes-Machine-Learning-Data/")

# load data (unchanged as provided)
synthetic_df = pd.read_csv("data/synthetic_data.csv")
real_df = pd.read_csv("data/real_data.csv")
features_description = pd.read_csv("data/feature_descriptions.csv")

# print the first few rows of the data set (unchanged as provided)
with pd.option_context('display.max_columns', None, 
                       'display.expand_frame_repr', False):  # configure pandas to print all columns
    print("Displaying the first five rows of the *synthetic* data set:\n")
    print(synthetic_df.head(5))
    print("The full size of the synthetic data set is", synthetic_df.shape)
    print("\n Displaying the first five rows of the *real* data set:\n")
    print(real_df.head(5))
    print("The full size of the real data set is", synthetic_df.shape)
    print("\n The features are described as the following: \n")
    print(features_description.to_csv(index=False))

### preprocess the data for machine learning

# y: readmitted
# substitute "NO" with 2, "<30" with 0, ">30" with 1
# real_df.readmitted = real_df.readmitted.replace(["NO", "<30", ">30"], [2, 0, 1])
real_df.readmitted.value_counts()
le_real = LabelEncoder()
real_y = le_real.fit_transform(real_df.readmitted)  # numpy.ndarray
le_real.inverse_transform(real_y)  # maybe for visualisation

synthetic_df.readmitted.value_counts()
le_synthetic = LabelEncoder()
synthetic_y = le_synthetic.fit_transform(synthetic_df.readmitted)  # numpy.ndarray
le_synthetic.inverse_transform(synthetic_y)  # maybe for visualisation

# x
real_df = real_df.drop(labels="readmitted", axis=1)  # axis 1 means columns
real_x = pd.get_dummies(real_df)  # one-hot encode, non-categorical variables will be left unchanged
real_x_columns = real_x.columns  # maybe for visualisation

synthetic_df = synthetic_df.drop(labels="readmitted", axis=1)  # axis 1 means columns
synthetic_x = pd.get_dummies(synthetic_df)  # non-categorical variables will be left unchanged
synthetic_x_columns = synthetic_x.columns  # maybe for visualisation

# only keep common columns
common_col = real_x_columns[real_x_columns.isin(synthetic_x_columns)]
synthetic_x = synthetic_x[common_col]
real_x = real_x[common_col]

# convert into numpy array, not sure whether it is necessary
real_x = np.array(real_x)
synthetic_x = np.array(synthetic_x)

# also have a look at the data processed by teammates
haseeb_synthetic = pd.read_csv("Haseeb/synthetic_data_dum1.csv")
haseeb_real = pd.read_csv("Haseeb/real_data_dum1.csv")

### random forest
# train a model on synthetic data to predict hospital readmission
rfc = RandomForestClassifier(n_estimators=1000).fit(synthetic_x, synthetic_y)

# test the model on real data
rfc_accuracy = accuracy_score(real_y, rfc.predict(real_x))  # only 52% accuracy;
# NOTE: or only use important variables

# train and test model on only the real data
half_sample_size = round(len(real_y) / 2)
real_y_train = real_y[0:half_sample_size]
real_y_test = real_y[half_sample_size : len(real_y)]  # begin at : end before : step size(default 1), 0-indexed
real_x_train = real_x[0:half_sample_size, :]
real_x_test = real_x[half_sample_size : len(real_y), :]

rfc_real = RandomForestClassifier(n_estimators=1000).fit(real_x_train, real_y_train)
rfc_accuracy_real = accuracy_score(real_y_test, rfc_real.predict(real_x_test))  # also merely 56%


### k nearest neighbour classification

# find the best value of k using cross-validation


### logistic regression

lrm = LogisticRegression(max_iter=1000).fit(synthetic_x, synthetic_y)
lrm_accuracy = accuracy_score(real_y, lrm.predict(real_x))
# NOTE: STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.