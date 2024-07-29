import time

import pandas as pd

import XAI_Swarm_Opt
from XAI import XAI
from colorama import Style, Fore
Data = pd.read_csv('FIFA 2018 Statistics.csv')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# grabs the names of features
features_name = [i for i in Data.columns if Data[i].dtype == np.int64]

# creates the dataset
X = Data[features_name]
y = Data['Man of the Match'] == 'Yes'

# splits the data into training and test set
x_train, x_val, y_train, y_val = train_test_split(X, y)

# print(x_val.head())

# run A MODEL for testing
model = RandomForestClassifier().fit(x_train, y_train)


row_number = 2
sample = x_val.iloc[row_number]
sample_y = y_val.iloc[row_number]
sample_y = 1 if sample_y == True else 0
sample = sample.values.reshape(1, -1)



# printing stuff
print(Style.BRIGHT + Fore.LIGHTCYAN_EX + 'FIFA2018 Dataset:')
print(Style.BRIGHT + Fore.CYAN + 'X: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample))
print(Style.BRIGHT + Fore.CYAN + 'y: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample_y),'\n')
print(Style.BRIGHT + Fore.CYAN + 'Blackbox model prediction: ', Style.BRIGHT + Fore.YELLOW + str(model.predict_proba(sample)),'\n')


# creates an of zeros 
S = np.zeros(np.size(sample[0]))
for i in range(np.size(sample[0])):
    S[i] = sample[0][i]

# def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up):
# A = XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 4000, -1, 1, features_name).XAI_swarm_Invoke()

temp_categorical = {"Begin_Categorical": 5, "Categorical_Index": [1, 2]}

for i in range(5):
    A = XAI_Swarm_Opt.XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 20,30, -1, 1, features_name, temp_categorical, True).XAI_swarm_Invoke()


print("\n\n\nPast this point \n\n\n")

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
num_features = np.size(S)
t1 = time.time_ns()
explainer = lime.lime_tabular.LimeTabularExplainer(x_train.values)
explainer = explainer.explain_instance(S,model.predict_proba, num_features = num_features)
t2 = time.time_ns()

print(explainer.local_pred)
print('lime time',(t2 - t1) * 10.0**-9)