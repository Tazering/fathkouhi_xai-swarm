"""
This python file stores algorithms for running shap and explanations
from base XAI algorithms.

Mostly adapted from: 

github.com/EmmanuelDoumard/local_explanation_comparative_study

Further details in the original github repository, but each function is applied to a specific explanation approach.
For convenience, I have listed the order below:

LIME
kernelSHAP
SpearMan
Complete
TreeSHAP
TreeSHAP_Approx
"""

import pandas as pd
import numpy as np

import time
import numpy as np
import lime.lime_tabular as lime_tabular
import shap
from tqdm.auto import tqdm

import coalitional_methods as coal
import complete_method as cmpl

import helpful_utils.data_tools as data_tools
import xgboost as xgb

# LIME
def explanation_values_lime(X, clf, mode, look_at=1, num_samples=5000, silent=False):
    t0 = time.time()

    explainer = lime_tabular.LimeTabularExplainer(training_data=X.values, mode = mode, feature_names=X.columns)

    # explainer = lime_tabular.LimeTabularExplainer(X.values,mode=mode,feature_names=X.columns)
    if silent:
        inf_values = np.array([[v for (k,v) in sorted(explainer.explain_instance(X.values[i],clf.predict_proba,labels=(look_at,), num_samples=num_samples,num_features=X.shape[1]).as_map()[look_at])] for i in range(X.shape[0])])
    else:
        inf_values = np.array([[v for (k,v) in sorted(explainer.explain_instance(X.values[i],clf.predict_proba,labels=(look_at,), num_samples=num_samples,num_features=X.shape[1]).as_map()[look_at])] for i in tqdm(range(X.shape[0]))])
    t1 = time.time()
        
    # Generate explanation compatible with shap
    explanation = shap.Explanation(inf_values,
                                base_values=np.zeros(X.shape[0]),
                                data=X.values,
                                feature_names=X.columns.to_list())

    shap_values = pd.DataFrame(inf_values,
                            columns=X.columns)
        
    return explanation, shap_values, t1-t0

# kernelshap: works for now
def explanation_values_kernelSHAP(X, clf, n_background_samples=None, look_at=1, silent=False):
    t0 = time.time()
    if n_background_samples:
        if n_background_samples < X.shape[0]:
            explainer = shap.KernelExplainer(clf.predict_proba, shap.kmeans(X,n_background_samples))
        else:
            explainer = shap.KernelExplainer(clf.predict_proba, X)
    else:
        explainer = shap.KernelExplainer(clf.predict_proba, X)

    expected_value = explainer.expected_value[look_at]
    shap_values = explainer.shap_values(X, silent=True)

    t1 = time.time()
    
    explanation = generate_correct_explanation(shap_values=shap_values[look_at],
                                               base_values=expected_value*np.ones(X.shape[0]),
                                               X=X,
                                               look_at=look_at)


    # shap_values = pd.DataFrame(shap_values[look_at],
    #                            index=X.columns)
    
    
    
    return explanation, shap_values, t1-t0

# spearman values
def explanation_values_spearman(X, y, clf, rate, problem_type, complexity=False, fvoid=None, look_at=1, progression_bar=True):
    t0 = time.time()
    spearman_inf, _, _ = coal.coalitional_method(X, y, clf, rate, problem_type, fvoid=fvoid, complexity=complexity, method='spearman', look_at=look_at, progression_bar=progression_bar)
    t1 = time.time()
    
    if fvoid is None:
        if problem_type == "Classification":
            fvoid = (
                y.value_counts(normalize=True).sort_index().values
            )
        elif problem_type == "Regression":
            fvoid = y.mean()
    

    explanation = generate_correct_explanation(shap_values=spearman_inf.values,
                                               base_values=fvoid[look_at]*np.ones(X.shape[0]),
                                               X=X,
                                               look_at=look_at)
    
    return explanation, spearman_inf, t1-t0

#  complete approach
def explanation_values_complete(X, y, clf, problem_type, fvoid=None, look_at=1, progression_bar=True):
    t0 = time.time()
    complete_inf, _ = cmpl.complete_method(X, y, clf, "Classification", fvoid=fvoid, look_at=look_at, progression_bar=progression_bar)
    t1 = time.time()
    
    if fvoid is None:
        if problem_type == "Classification":
            fvoid = (
                y.value_counts(normalize=True).sort_index().values
            )
        elif problem_type == "Regression":
            fvoid = y.mean()
            
    explanation = generate_correct_explanation(shap_values=complete_inf.values,
                                               base_values=fvoid[look_at]*np.ones(X.shape[0]),
                                               X=X,
                                               look_at=look_at)
    
    return explanation, complete_inf, t1-t0

# TREE_SHAP
def explanation_values_treeSHAP(X, clf, look_at=1):
    t0 = time.time()
    explainer = shap.TreeExplainer(clf, X, model_output = "probability")
    explanation = explainer(X, check_additivity=False)
    t1 = time.time()
    explanation = generate_correct_explanation(explanation,look_at=look_at)
    shap_values = pd.DataFrame(explanation.values,
                               columns=X.columns)
    
    return explanation, shap_values, t1-t0

#   TREE_SHAP_APPROX
def explanation_values_treeSHAP_approx(X, clf, look_at=1):
    t0 = time.time()
    explainer = shap.TreeExplainer(clf)
    if len(explainer.expected_value) > 1:
        expected_value = explainer.expected_value[look_at]
    else: #SHAP NE SORT QU'UNE VALEUR POUR UNE PREDICTION BINAIRE AVEC XGBOOST ON SAIT PAS POURQUOI SUPER
        expected_value = explainer.expected_value[0]
    shap_values = explainer.shap_values(X,approximate=True)
    t1 = time.time()
    if type(shap_values)==list:
        shap_values_expl = shap_values[look_at]
    else:
        shap_values_expl = shap_values
    explanation = generate_correct_explanation(shap_values=shap_values_expl,
                                               base_values=expected_value*np.ones(X.shape[0]),
                                               X=X,
                                               look_at=look_at)
    shap_values = pd.DataFrame(explanation.values,
                               columns=X.columns)
    
    return explanation, shap_values, t1-t0


# generates the correct shep explanations
def generate_correct_explanation(explanation=None,shap_values=None,base_values=None,X=None,look_at=1):
    # Recreate a shap explanation object that can be used for Waterfall plots
    
    if explanation is None:
        if (shap_values is None) | (base_values is None) | (X is None):
            raise Exception("If you pass no explanation, you need to pass shap_values, base_values and X to construct an Explanation object")
            
    if shap_values is None:
        if len(np.array(explanation.values).shape) == 3:
            shap_values = explanation.values[:,:,look_at]
        else:
            shap_values = explanation.values
    if base_values is None:
        if len(np.array(explanation.base_values).shape) == 2:
            base_values = explanation.base_values[:,look_at]
        else:
            base_values = explanation.base_values
    if X is None:
        X = pd.DataFrame(explanation.data,columns=explanation.feature_names)
        
    correct_explanation = shap.Explanation(shap_values,
                                         base_values=base_values,
                                         data=X.values,
                                         feature_names=X.columns.to_list())

    return correct_explanation

# special cases functions
# The two following methods transform the output of a multiclass prediction into a "one versus all" one
# Used only to compute XGBoost classifier output so that it is comparable to other models output
    
def treeSHAP_multi_to_binary_class(X, y, look_at=1):
    shap_values=[]
    base_values=[]
    t0 = time.time()
    for distinct_class in y.sort_values().unique():
        y_bin = y.copy()
        y_bin.loc[y_bin!=distinct_class] = -1 #valeur temporaire
        y_bin.loc[y_bin==distinct_class] = 1
        y_bin.loc[y_bin==-1] = 0
        
        xgbc_bin = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1)
        xgbc_bin.fit(X, y_bin)
        explainer = shap.TreeExplainer(xgbc_bin, X, model_output = "probability")
        shap_values_bin = explainer.shap_values(X)
        shap_values.append(shap_values_bin)
        base_values.append(explainer.expected_value)
    t1 = time.time()
    
    shap_values = pd.DataFrame(shap_values[look_at],columns=X.columns)
    explanation = generate_correct_explanation(shap_values=shap_values,
                                               base_values=base_values,
                                               X=X,
                                               look_at=look_at)
    
    return explanation, shap_values, t1-t0


def treeSHAPapprox_multi_to_binary_class(X, y, look_at=1):
    shap_values=[]
    base_values=[]
    t0 = time.time()
    for distinct_class in y.sort_values().unique():
        y_bin = y.copy()
        y_bin.loc[y_bin!=distinct_class] = -1 #tmp
        y_bin.loc[y_bin==distinct_class] = 1
        y_bin.loc[y_bin==-1] = 0
        
        xgbc_bin = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_jobs=1)
        xgbc_bin.fit(X, y_bin)
        explainer = shap.TreeExplainer(xgbc_bin, data=None)
        shap_values_bin = explainer.shap_values(X)
        shap_values.append(shap_values_bin)
        base_values.append(explainer.expected_value)
    t1 = time.time()
    
    shap_values = pd.DataFrame(shap_values[look_at],columns=X.columns)
    explanation = generate_correct_explanation(shap_values=shap_values,
                                               base_values=base_values,
                                               X=X,
                                               look_at=look_at)
    
    return explanation, shap_values, t1-t0