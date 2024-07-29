import time
import xgboost
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score

import XAI_Swarm_Opt
from XAI import XAI
import numpy as np
from colorama import Style, Fore
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

Data = pd.read_csv('C:/Users/ASUS/Documents/Windsor_XAI/breast-cancer.csv')
features_name = Data.columns[2:]
X = Data[features_name].to_numpy()
Y = Data[Data.columns[1]].tolist()
# Y = Data[Data.columns[1]]
# print(Data[Data.columns[1]][0])
# print(Data[Data.columns[1]][568])
# Y = Y.astype('category').cat.codes
# print(Y)

for i in range(len(Y)):
    if Y[i] == 'M':
        Y[i] = 1
    else:
        Y[i] = 0

x_train, x_test, y_train, y_test = train_test_split(X,Y)

# model = RandomForestClassifier(n_estimators=10, random_state=0).fit(x_train, y_train)
model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(x_train, y_train)
# model = xgboost.XGBClassifier().fit(x_train, y_train)
# model = KNeighborsClassifier(n_neighbors=5).fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_train = [round(value) for value in y_pred_train]
print('train accuracy: ', accuracy_score(y_train,y_pred_train))

y_pred_test = model.predict(x_test)
y_pred_test = [round(value) for value in y_pred_test]
print('test accuracy: ', accuracy_score(y_test,y_pred_test))


sample = np.array([[1.127e+01, 1.550e+01, 7.338e+01, 3.920e+02, 8.365e-02, 1.114e-01, 1.007e-01,
  2.757e-02, 1.810e-01, 7.252e-02, 3.305e-01, 1.067e+00, 2.569e+00, 2.297e+01,
  1.038e-02, 6.669e-02, 9.472e-02, 2.047e-02, 1.219e-02, 1.233e-02, 1.204e+01,
  1.893e+01, 7.973e+01, 4.500e+02, 1.102e-01, 2.809e-01, 3.021e-01, 8.272e-02,
  2.157e-01, 1.043e-01]])
# sample_y = y_test[row_number]
sample_y = 0

sample = sample.reshape(1, -1)

print(Style.BRIGHT + Fore.LIGHTCYAN_EX + 'Breast Cancer Dataset:')
print(Style.BRIGHT + Fore.CYAN + 'X: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample))
print(Style.BRIGHT + Fore.CYAN + 'y: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample_y),'\n')
print(Style.BRIGHT + Fore.CYAN + 'Blackbox model prediction: ', Style.BRIGHT + Fore.YELLOW + str(model.predict_proba(sample)),'\n')

S = np.zeros(np.size(sample[0]))
for i in range(np.size(sample[0])):
    S[i] = sample[0][i]

# print(model.predict_proba(sample)[1])
# def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up):
# A = XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 4000, -1, 1, features_name).XAI_swarm_Invoke()
# for i in range(5):

# for i in range(3):
#     A = XAI_Swarm_Opt.XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 20,30, -1, 1, features_name).XAI_swarm_Invoke()

print(model.predict_proba(sample)[0][1])
for i in range(3):
    A = XAI_Swarm_Opt.XAI(model.predict_proba(sample)[0][1], S, np.size(S), 50, 20,30, -1, 1, features_name.values.tolist(), None, False).XAI_swarm_Invoke()
# print()

import shap
# t1 = time.time_ns()
# explainer = shap.TreeExplainer(model)
# dd = explainer.shap_values(sample)
# t2 = time.time_ns()
# print('Tree shap time: ', (t2 - t1) * 10.0**-9)
# print('Tree Explainer', np.abs(model.predict_proba(sample)[0][1] - explainer.expected_value[1] - np.sum(dd[1])))


import lime.lime_tabular
num_features = np.size(S)
t1 = time.time_ns()
explainer = lime.lime_tabular.LimeTabularExplainer(x_train)
explainer = explainer.explain_instance(S,model.predict_proba, num_features = num_features)
t2 = time.time_ns()
print('Lime Explainer', np.abs(model.predict_proba(sample)[0][1] - explainer.intercept[1] - sum([weight[1] for weight in explainer.local_exp[1]])))
print('lime time',(t2 - t1) * 10.0**-9)


t1 = time.time_ns()
explainer = shap.KernelExplainer(model.predict_proba,x_train)
dd = explainer.shap_values(sample)
t2 = time.time_ns()
print('Kernel shap time: ', (t2 - t1) * 10.0**-9)
print('Kernel Explainer', np.abs(model.predict_proba(sample)[0][1] - explainer.expected_value[1] - np.sum(dd[1])))


t1 = time.time_ns()
explainer = shap.Explainer(model.predict_proba,x_train)
dd = explainer(sample)
t2 = time.time_ns()
print('Shap time: ', (t2 - t1) * 10.0**-9)
print('Shap Explainer', np.abs(model.predict_proba(sample)[0][1] - dd.base_values[0][1] - dd.values.sum(axis = 1)[0][1]))
