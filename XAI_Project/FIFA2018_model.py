import time

import pandas as pd

import XAI_Swarm_Opt
from XAI import XAI
from colorama import Style, Fore

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import data_tools as data_tools

#############################################
#   FETCHING THE DATA
#############################################

# grab the data
fifa_data = pd.read_csv('FIFA 2018 Statistics.csv')

# describe the data
print(fifa_data.describe)
print("\n")

# grabs the names of numerical features
numerical_features = [i for i in fifa_data.columns if fifa_data[i].dtype == np.int64]

# print the features
data_tools.print_variable("numerical_features", numerical_features)

# creates the dataset
X = fifa_data[numerical_features]
y = fifa_data['Man of the Match'] == 'Yes'

# splits the data into training and test set
x_train, x_val, y_train, y_val = train_test_split(X, y)

###################################################
#   RUN A MODEL
###################################################

# run A MODEL for testing
model = RandomForestClassifier().fit(x_train, y_train)

###################################################
#   GRAB A SAMPLE
###################################################

# grab a sample 
sample_number = 2

data_tools.print_variable("x_val", x_val)

# details of a single datapoint
sample = x_val.iloc[sample_number]
sample_y = y_val.iloc[sample_number]
sample_y = 1 if sample_y == True else 0
sample = sample.values.reshape(1, -1)

data_tools.print_variable("sample", sample)

# printing stuff
data_tools.print_dataset_sample("FIFA2018 Dataset", sample, sample_y, model)

# converts sample to a single list
sample_size = np.size(sample[0])

# S = np.zeros(sample_size)
# for i in range(sample_size):
#     S[i] = sample[0][i]
# data_tools.print_variable("S", S)

sample_list = sample[0]

data_tools.print_variable("sample_list", sample_list)

###################################################
#   RUN XAI
###################################################

# def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up):
# A = XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 4000, -1, 1, features_name).XAI_swarm_Invoke()

temp_categorical = {"Begin_Categorical": 5, "Categorical_Index": [1, 2]}

for i in range(3):
    A = XAI_Swarm_Opt.XAI(max(model.predict_proba(sample)[0]), sample_list, np.size(sample_list), 50, 20,30, -1, 1, numerical_features, temp_categorical, False).XAI_swarm_Invoke()

# calculate times with xai
import shap
t1 = time.time_ns()

# use kernel shap
explainer = shap.KernelExplainer(model.predict_proba,x_train)
dd = explainer.shap_values(sample)
t2 = time.time_ns()
print('Kernel shap time: ', (t2 - t1) * 10.0**-9)

# use tree explainer
t1 = time.time_ns()
explainer = shap.TreeExplainer(model)
dd = explainer.shap_values(sample)
t2 = time.time_ns()
print('Tree shap time: ', (t2 - t1) * 10.0**-9)

# use lime
import lime.lime_tabular
num_features = np.size(sample_list)
t1 = time.time_ns()
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values)
explainer = explainer.explain_instance(sample_list,model.predict_proba, num_features = num_features)
t2 = time.time_ns()

print(explainer.local_pred)
print('lime time',(t2 - t1) * 10.0**-9)