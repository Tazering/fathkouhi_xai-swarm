import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

import XAI_Swarm_Opt 

df = pd.read_csv('ENB2012_data.csv')
X = df.drop(columns=['Y1', 'Y2'], axis=0)
Y = df[['Y1', 'Y2']]
X = np.array(X)
Y = np.array(Y)

x_train, x_test, y_train, y_test = train_test_split(X,Y)

# model = RandomForestRegressor(n_estimators=10, random_state=0).fit(x_train, y_train)
model = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(x_train, y_train)

S = x_test[0].reshape(1, -1)
sample = S
pred = model.predict(sample)
print(mean_squared_error(model.predict(x_test), y_test, squared=False))

# print(model.predict_proba(sample)[1])
# def __init__(self, model_predict, sample, size, no_pso, no_iteration, lb, up):
# A = XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 4000, -1, 1, features_name).XAI_swarm_Invoke()
# for i in range(5):

# for i in range(3):
#     A = XAI_Swarm_Opt.XAI(max(model.predict_proba(sample)[0]), S, np.size(S), 50, 20,30, -1, 1, features_name).XAI_swarm_Invoke()

for i in range(3):
    A = XAI_Swarm_Opt.XAI(model.predict(sample)[0][1], sample[0], np.size(S), 50, 20,30, -1, 1, df.columns.tolist(), None, False).XAI_swarm_Invoke()

import shap
import lime.lime_tabular
num_features = np.size(S)
t1 = time.time_ns()
explainer = lime.lime_tabular.LimeTabularExplainer(x_train, mode='regression')
explainer = explainer.explain_instance(x_test[0], model.predict, num_features = num_features)
t2 = time.time_ns()
print('Lime Explainer', np.abs(pred[0][1] - explainer.intercept[1] - sum([weight[1] for weight in explainer.local_exp[1]])))
print('lime time',(t2 - t1) * 10.0**-9)

t1 = time.time_ns()
explainer = shap.KernelExplainer(model.predict, data=x_train)
dd = explainer.shap_values(sample)
t2 = time.time_ns()
print('Kernel shap time: ', (t2 - t1) * 10.0**-9)
print('Kernel Explainer', np.abs(pred[0][1] - explainer.expected_value[1] - np.sum(dd[1])))


t1 = time.time_ns()
explainer = shap.Explainer(model.predict,x_train)
dd = explainer(sample)

t2 = time.time_ns()
print('Shap Explainer', np.abs(pred[0][1] - dd.base_values[0][1] - dd.values.sum(axis = 1)[0][1]))
print('Shap time: ', (t2 - t1) * 10.0**-9)
