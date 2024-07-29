import time

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import XAI_Swarm_Opt
from XAI import XAI
import numpy as np
from colorama import Style, Fore
from sklearn.ensemble import RandomForestClassifier

Data_Train = pd.read_csv('D:/Airline_Satisfaction/train.csv')
Data_Test = pd.read_csv('D:/Airline_Satisfaction/test.csv')

Categorical_Features = ['Gender','Customer Type','Type of Travel','Class']

Data_Train = Data_Train.drop(['Unnamed: 0', 'id'], axis=1)
Data_Test = Data_Test.drop(['Unnamed: 0', 'id'], axis=1)

Data_Train['Arrival Delay in Minutes'].fillna(Data_Train['Arrival Delay in Minutes'].max(), inplace=True)
Data_Test['Arrival Delay in Minutes'].fillna(Data_Test['Arrival Delay in Minutes'].max(), inplace=True)


Categorical_Index = []

for i in Categorical_Features:
    Categorical_Index.append(len(Data_Train[i].unique()))

le = OneHotEncoder()
X_Encoded = le.fit_transform(Data_Train[['Gender','Customer Type','Type of Travel','Class']])
Y_Encoded = le.fit_transform(Data_Train[['satisfaction']])
Data_Train = Data_Train.drop(['Gender','Customer Type','Type of Travel','Class', 'satisfaction'], axis = 1)
X_Train = np.append(Data_Train.values.tolist(), list(X_Encoded.toarray()), axis = 1)
Y_Train = Y_Encoded.toarray()

le = OneHotEncoder()
X_Encoded = le.fit_transform(Data_Test[['Gender','Customer Type','Type of Travel','Class']])
Y_Encoded = le.fit_transform(Data_Test[['satisfaction']])
Data_Test = Data_Test.drop(['Gender','Customer Type','Type of Travel','Class', 'satisfaction'], axis = 1)
X_Test = np.append(Data_Test.values.tolist(), list(X_Encoded.toarray()), axis = 1)
Y_Test = Y_Encoded.toarray()

Begin_Categorical = len(Data_Test.columns)
features_name = list(Data_Train.columns.values) + Categorical_Features
Categorical = {'Categorical_Index': Categorical_Index, 'Categorical_Features' : Categorical_Features, 'Begin_Categorical': Begin_Categorical}




model = RandomForestClassifier().fit(X_Train, Y_Train)
# row_number = np.random.randint(len(x_test))
row_number = 2
sample = X_Test[row_number]
sample_y = Y_Test[row_number]
sample = sample.reshape(1, -1)

print(Style.BRIGHT + Fore.LIGHTCYAN_EX + 'Airline Satisfaction Dataset:')
print(Style.BRIGHT + Fore.CYAN + 'X: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample))
print(Style.BRIGHT + Fore.CYAN + 'y: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample_y),'\n')
print(Style.BRIGHT + Fore.CYAN + 'Blackbox model prediction: ', Style.BRIGHT + Fore.YELLOW + str(model.predict_proba(sample)[1]),'\n')

S = sample[0]
# # def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up):
# # A = XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 4000, -1, 1, features_name).XAI_swarm_Invoke()
# A = XAI_Swarm_Opt.XAI(model.predict_proba(sample)[1][0], S, np.size(S), 50, 20,30, -1, 1, features_name).XAI_swarm_Invoke()

A = XAI_Swarm_Opt.XAI(model.predict_proba(sample)[0][0][1], S, np.size(S), 50, 20,30, -1, 1, features_name, Categorical,Categorical_Status= True).XAI_swarm_Invoke()









# # for i in range(5):
# #     A = XAI_Swarm_Opt.XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 20,30, -1, 1, features_name).XAI_swarm_Invoke()
#
# import shap
# # t1 = time.time_ns()
# # explainer = shap.KernelExplainer(model.predict_proba,x_train)
# # dd = explainer.shap_values(sample)
# # t2 = time.time_ns()
# # print('Kernel shap time: ', (t2 - t1) * 10.0**-9)
# #
# # t1 = time.time_ns()
# # explainer = shap.TreeExplainer(model)
# # dd = explainer.shap_values(sample)
# # t2 = time.time_ns()
# # print('Tree shap time: ', (t2 - t1) * 10.0**-9)
# #
# #
# # import lime.lime_tabular
# # num_features = np.size(S)
# # t1 = time.time_ns()
# # explainer = lime.lime_tabular.LimeTabularExplainer(x_train)
# # explainer = explainer.explain_instance(S,model.predict_proba, num_features = num_features)
# # t2 = time.time_ns()
# #
# # print(explainer.local_pred)
# # print('lime time',(t2 - t1) * 10.0**-9)