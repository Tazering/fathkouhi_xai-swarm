#Imports
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import make_column_selector as selector
from arff import ArffException

import openml as oml

import sklearn
from lime import lime_tabular
import shap

import pickle
import time
from tqdm.auto import tqdm

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#Local imports
import coalitional_methods as coal
import complete_method as cmpl

import warnings
warnings.simplefilter("ignore",category=UserWarning) # Silence user warnings during model training

# Colormap definition
import plotly.express.colors as px_colors
colormap = px_colors.qualitative.Plotly
colormap[5] = "#FF00DF"
colormap[0] = "#056FAB"

# Matplotlib fontsize adjustments
params = {'legend.fontsize': 22,
          'figure.figsize': (15, 5),
         'axes.labelsize': 26,
         'axes.titlesize': 26,
         'xtick.labelsize': 24,
         'ytick.labelsize': 24}
mpl.rcParams.update(params)
#############################
#### Explanation methods ####
#############################

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


# The following functions create an explanation object and compute influence values as a DataFrame for the methods of interest

def explanation_values_treeSHAP(X, clf, look_at=1):
    t0 = time.time()
    explainer = shap.TreeExplainer(clf, X, model_output = "probability")
    explanation = explainer(X, check_additivity=False)
    t1 = time.time()
    explanation = generate_correct_explanation(explanation,look_at=look_at)
    shap_values = pd.DataFrame(explanation.values,
                               columns=X.columns)
    
    return explanation, shap_values, t1-t0


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

    shap_values = pd.DataFrame(shap_values[look_at],
                               columns=X.columns)
    
    return explanation, shap_values, t1-t0

def explanation_values_spearman(X, y, clf, rate, problem_type, complexity=False, fvoid=None, look_at=1, progression_bar=True):
    t0 = time.time()
    spearman_inf = coal.coalitional_method(X, y, clf, rate, problem_type, fvoid=fvoid, complexity=complexity, method='spearman', look_at=look_at, progression_bar=progression_bar)
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
    
def explanation_values_complete(X, y, clf, problem_type, fvoid=None, look_at=1, progression_bar=True):
    t0 = time.time()
    complete_inf = cmpl.complete_method(X, y, clf, "Classification", fvoid=fvoid, look_at=look_at, progression_bar=progression_bar)
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

def explanation_values_lime(X, clf, mode, look_at=1, num_samples=5000, silent=False):
    t0 = time.time()
    explainer = lime_tabular.LimeTabularExplainer(X.values,mode=mode,feature_names=X.columns)
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

# The two following methods transform the output of a multiclass prediction into a "one versus all" one
# Used only to compute XGBoost classifier output so that it is comparable to other models output
    
def treeSHAP_multi_to_binary_class(X, y, xgbc, look_at=1):
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


def treeSHAPapprox_multi_to_binary_class(X, y, xgbc, look_at=1):
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

#########################
###### EXPERIMENTS ######
#########################

def process_dataset_openml(dataset_id, variable_pred) :
    """
    Retrieves dataset from openml based on the task id in parameters, 
    and pre-process data.
    Preprocessing follow these step :
        - for numerical attributs : replace none value by the median & scale the data
        - for categorical attributs : replace none value by 'missing' & Encode categorical features as an integer array
        - for labels : encode target labels with value between 0 and n_classes-1
    
    Parameters
    ----------
    dataset_id : int
        Index of the dataset in Open ML.
    variable_pred : string
        Name to use for the labels target.

    Returns
    -------
    X_process : pandas.DataFrame
        Processed datas from the dataset.
    y_process : pandas.DataFrame
        Processed target from the dataset.
    """
    
    try:
        dataset = oml.datasets.get_dataset(dataset_id)
    except oml.exceptions.OpenMLServerException as e:
        # Error code for "no qualities found"
        if e.code == 362:
            print(e)
        return None, None
    
    try:
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )
    except ArffException as e:
        print(e)
        return None, None
    
    X = X.select_dtypes(exclude=['object'])
    
    numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OrdinalEncoder())
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=["category"])),
        ('cat', categorical_transformer, selector(dtype_include="category"))
    ])
    
    X_process = preprocessor.fit_transform(X)
    X_process = pd.DataFrame(X_process,columns=X.select_dtypes(exclude=["category"]).columns.append(X.select_dtypes(include=["category"]).columns)) #Order
    
    label_encoder = LabelEncoder()
    y_process = label_encoder.fit_transform(y)
    y_process = pd.Series(y_process, name = variable_pred)
    
    return X_process, y_process


# Returns a DataFrame with a filtered list of datasets from OpenML
def get_datasets():
    df_datasets = oml.datasets.list_datasets(output_format="dataframe")
    df_datasets = df_datasets.drop_duplicates(subset="did").drop_duplicates(subset=["name"]).drop_duplicates(subset=df_datasets.drop(["did","name","version","uploader"],axis=1).columns, keep="last")
    df_datasets = df_datasets.loc[(df_datasets["NumberOfFeatures"]<15) & #OpenML counts the target in the number of features, we want up to 13 features
                                  (df_datasets["NumberOfFeatures"]>0) &
                                  (df_datasets["NumberOfInstances"]<10000) &
                                  (df_datasets["NumberOfInstances"]>50) &
                                  (df_datasets["NumberOfClasses"]>1) &
                                  (df_datasets["NumberOfClasses"]<1000)]  #Some strange dataset has 3169 classes
    
    return df_datasets
    
# Compute the explanation of a model, given the method, the model and the dataset
# Add the results with the time needed to compute them to a dict
def test_and_store(dict_results, did, X, clf, model_name, explanation_funct, method_name, **kwargs):
    explanation, inf, t = explanation_funct(X=X, clf=clf, **kwargs)
    if model_name not in dict_results[did]:
        dict_results[did][model_name] = {}
    dict_results[did][model_name].update({method_name:
                                          {"explanation" : explanation,
                                           "inf" : inf,
                                           "time" : t}
                                         })

# Run all the studied methods (LIME, SHAP, Complete and Spearman) on a single dataset and single trained model 
def run_all_tests(dict_results, did, X, y, clf, model_name, rate, mode, problem_type, complexity=True, fvoid=None, look_at=1, n_background_samples=None):
    if did not in dict_results: #add dataset to dict
        dict_results[did] = {"X":X, "y": y}
    if ("X" not in dict_results[did]) | ("y" not in dict_results[did]):
        dict_results[did]["X"] = X
        dict_results[did]["y"] = y
        
    test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_complete, method_name="complete", y=y, problem_type=problem_type, fvoid=fvoid, look_at=look_at) #Complete
    test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_spearman, method_name="spearman"+str(rate), y=y, rate=rate, problem_type=problem_type, complexity=True, fvoid=fvoid, look_at=look_at) #Spearman
    test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_lime, method_name="LIME", mode=mode, look_at=look_at, num_samples=100) #LIME
    try:
        test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_treeSHAP_approx, method_name="treeSHAPapprox", look_at=look_at) #SHAP
        test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_treeSHAP, method_name="treeSHAP", look_at=look_at) #SHAP
    except:
        print("Model not suited for TreeSHAP. Testing with kernel explainer with {} background samples".format(n_background_samples))
        try:
            test_and_store(dict_results, did, X, clf, model_name, explanation_funct=explanation_values_kernelSHAP, method_name="kernelSHAP"+(str(n_background_samples) if n_background_samples else ""), look_at=look_at,n_background_samples=n_background_samples) #SHAP
        except:
            print("kernelSHAP{} not working for this dataset".format(str(n_background_samples) if n_background_samples else ""))

# Run all methods with all models given in parameters on all given datasets
# Save all results to a dictionary which can serve as checkpoint
def study_all_dataset(clf_dict, df_datasets=None, dict_results = {}, limit=60, n_background_samples=None):    
    if df_datasets is None:
        df_datasets = get_datasets()
        
    dataset_ids = df_datasets.sort_values("did")["did"]
    
    for n, did in enumerate(dataset_ids[:limit]):
        if did not in dict_results:
            print("--- Try ", n, "/",limit)
            print("--- Dataset number ", did, "---")
            variable_pred = "target"
            X, y = process_dataset_openml(did, variable_pred)
            
            if X is not None: #process_data_openml can return None
                print("Dataset shape :", X.shape)
                if X.shape[1] > 0:
                    df_datasets.loc[did,"NumberOfFeatures"] = X.shape[1]
                    display(df_datasets.loc[did,["did","name","NumberOfInstances","NumberOfFeatures","NumberOfClasses"]])

                    dict_results[did] = {"X":X, "y": y}

            else:
                print("Can't retrieve dataset number ",did)
        
        if did in dict_results:
            for clf in clf_dict:
                if clf not in dict_results[did]:
                    print("Testing {} on dataset {}".format(clf,did))
                    display(df_datasets.loc[did,["did","name","NumberOfInstances","NumberOfFeatures","NumberOfClasses"]])
                    
                    clf_to_train = sklearn.base.clone(clf_dict[clf])
                    if type(dict_results[did]["y"])==pd.DataFrame:
                        display(dict_results[did]["y"])
                        clf_to_train.fit(dict_results[did]["X"],dict_results[did]["y"]["target"])
                    else:
                        clf_to_train.fit(dict_results[did]["X"],dict_results[did]["y"])
                    run_all_tests(dict_results, did, dict_results[did]["X"], dict_results[did]["y"], clf=clf_to_train, model_name=clf, rate=0.25, mode="classification", problem_type="Classification", complexity=True, fvoid=None, look_at=1, n_background_samples=n_background_samples)
                    
                    pickle.dump(dict_results,open("pickles/dict_results.pkl", "wb" ))
                    
                    
###################################
####### Draw graphs methods #######
###################################
    
def summary_plots(list_shap_values, X, list_names=None):
    if list_names is None:
        list_names=[str(i) for i in range(len(list_shap_values))]
        
    for i,shap_value in enumerate(list_shap_values):
        plt.figure()
        shap.summary_plot(shap_value.values,X,show=False)
        plt.title(list_names[i])
        plt.show();
        
def summary_subplots(list_shap_values, X, list_names=None):
    if list_names is None:
        list_names=[str(i) for i in range(len(list_shap_values))]
    plt.figure(figsize=(30*len(list_shap_values),10))
    for i in range(len(list_shap_values)):
        plt.subplot(1, len(list_shap_values), i+1)
        shap.summary_plot(list_shap_values[i].values,X,show=False,plot_size=(20,8), color_bar=(i==len(list_shap_values)-1))
        plt.title(list_names[i], fontsize=params['axes.titlesize']-4)
        plt.xlabel("Influence values")
    plt.tight_layout()
    plt.show();
        
def sort_by_influence(df,ascending=True):
    return df.sort_index(key=lambda x:df.loc[:,x].abs().mean(),axis=1, ascending=ascending)

def partial_dependance_plots(list_shap_values,X,list_names=None, save_path=None):
    if list_names is None:
        list_names=[str(i) for i in range(len(list_shap_values))]
    
    for i,var in enumerate(sort_by_influence(list_shap_values[0],ascending=False).columns):
        fig, axes = plt.subplots(nrows=1, ncols=len(list_names), sharey=True, figsize=(4*len(list_shap_values),5))
        for j,shap_values in enumerate(list_shap_values):
            shap.dependence_plot(var,shap_values.values,X,interaction_index=None,show=False,x_jitter=0.5,alpha=1,ax=axes[j])
            axes[j].set_title(list_names[j], fontsize=params['axes.labelsize'])#,fontsize=18)
            axes[j].set_xlabel(var, fontsize=params['axes.labelsize']-6)
            axes[j].tick_params(axis='both', which='major', labelsize=params['xtick.labelsize']-4)
            if j>0:
                axes[j].set_ylabel(None)
                plt.setp(axes[j].get_yticklabels(), visible=False)
            else:
                axes[j].set_ylabel("Influence values for\n$\\bf{"+"\ ".join(var.split(" "))+"}$", fontsize=params['axes.labelsize'])
        plt.tight_layout()
        if save_path:
            Path(save_path+"dependence_plots/").mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path+"dependence_plots/"+"_".join(var.split(" "))+".jpg", bbox_inches='tight')
        plt.show()
    
def partial_dependance_subplots(list_shap_values,X,list_names=None):
    if list_names is None:
        list_names=[str(i) for i in range(len(list_shap_values))]
    
    fig, axes = plt.subplots(list_shap_values[0].shape[1],len(list_shap_values), sharey=True, figsize=(5*len(list_shap_values),6*list_shap_values[0].shape[1]))
    for i,var in enumerate(sort_by_influence(list_shap_values[0],ascending=False).columns):
        for j,shap_values in enumerate(list_shap_values):
            shap.dependence_plot(var,shap_values.values,X,interaction_index=None,show=False,x_jitter=0.5,alpha=1,ax=axes[i][j])
            axes[i][j].set_xlabel(list_names[j])#,fontsize=18)
            axes[i][j].tick_params(axis='both', which='major', labelsize=16)
            if j>0:
                axes[i][j].set_ylabel(None)
            else:
                axes[i][j].set_ylabel("Influence values for\n$\\bf{"+"\ ".join(var.split(" "))+"}$")#, fontsize=18)
        #axes[i].suptitle(var)
    plt.tight_layout()
        
def interaction_dependence_plots(list_shap_values, list_couple_vars, X, list_names=None, save_path=None):
    if list_names is None:
        list_names=[str(i) for i in range(len(list_shap_values))]
    
    for i in range(len(list_couple_vars)):
        fig, axes = plt.subplots(nrows=1, ncols=len(list_names), sharey=True, figsize=(4*len(list_shap_values),5))
        for j,shap_values in enumerate(list_shap_values):
            axes[j].scatter(shap_values.loc[:,list_couple_vars[i][0]],shap_values.loc[:,list_couple_vars[i][1]], s=16)
            #shap.dependence_plot(var,shap_values.values,X,interaction_index=None,show=False,x_jitter=0.5,alpha=1,ax=axes[j])
            axes[j].set_title(list_names[j], fontsize=params['axes.labelsize'])#,fontsize=18)
            axes[j].set_xlabel("Influence values for\n$\\bf{"+"\ ".join(list_couple_vars[i][0].split(" "))+"}$", fontsize=params['axes.labelsize']-2)
            axes[j].tick_params(axis='both', which='major', labelsize=params['xtick.labelsize']-4)
            if j>0:
                axes[j].set_ylabel(None)
                plt.setp(axes[j].get_yticklabels(), visible=False)
            else:
                axes[j].set_ylabel("Influence values for\n$\\bf{"+"\ ".join(list_couple_vars[i][1].split(" "))+"}$", fontsize=params['axes.labelsize'])
        plt.tight_layout()
        if save_path:
            Path(save_path+"interaction_plots/").mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path+"interaction_plots/"+"_".join(list_couple_vars[i][0].split(" "))+"-"+"_".join(list_couple_vars[i][1].split(" "))+".jpg", bbox_inches='tight')
        plt.show()
    
def single_method_analysis(dict_results, model_name, method_name, return_lists=False, show=True):
    time_method = []
    number_features = []
    number_instances = []
    list_did = []
    number_classes = []
    diff_with_complete=[]
    for did in dict_results:
        if model_name in dict_results[did]:
            if method_name in dict_results[did][model_name]:
                time_method.append(dict_results[did][model_name][method_name]["time"])
                diff_with_complete.append((dict_results[did][model_name]["complete"]["inf"] - dict_results[did][model_name][method_name]["inf"]).abs().mean().mean())
                number_features.append(dict_results[did]["X"].shape[1])
                number_instances.append(dict_results[did]["X"].shape[0])
                list_did.append(did)
                try:
                    number_classes.append(dict_results[did]["y"].iloc[:,0].unique().shape[0])
                except:
                    number_classes.append(dict_results[did]["y"].unique().shape[0])

    if show:
        plt.scatter(number_features,np.array(time_method)/np.array(number_instances))
        for i in range(len(time_method)):
            plt.text(number_features[i]+(plt.xlim()[1]-plt.xlim()[0])/100,time_method[i]/number_instances[i],list_did[i])
        plt.title("Execution time per instance of method {} by number of features".format(method_suffix))
        plt.xlabel("Number of features")
        plt.ylabel("Time per instance")
        plt.show()

        plt.scatter(number_instances,np.array(time_method)/np.array(number_instances))
        for i in range(len(time_method)):
            plt.text(number_instances[i]+(plt.xlim()[1]-plt.xlim()[0])/100,time_method[i]/number_instances[i],list_did[i])
        plt.title("Execution time per instance of method {} by number of instances".format(method_suffix))
        plt.xlabel("Number of instances")
        plt.ylabel("Time per instance")

        plt.show()

        df_stats = pd.DataFrame({"time":time_method,
                                 "time per instance":np.array(time_method)/np.array(number_instances),
                                 "number of features":number_features,
                                 "number of instances":number_instances,
                                 "dataset number":list_did,
                                 "number of classes":number_classes,
                                 "diff_with_complete":diff_with_complete})
        
        fig = px.scatter(df_stats, x="number of features", y="time per instance",
                         size=np.sqrt(df_stats["number of instances"]),size_max=20,
                         hover_data={"number of features":True, "number of instances":True, "number of classes":True, "time per instance":True, "dataset number":True, "diff_with_complete":True},
                         width=700,height=500)
        fig.show()
        
        fig = px.scatter(df_stats, x="number of features", y="diff_with_complete",
                         size=np.sqrt(df_stats["number of instances"]),size_max=20,
                         hover_data={"number of features":True, "number of instances":True, "number of classes":True, "time per instance":True, "dataset number":True, "diff_with_complete":True},
                         width=700,height=500)
        fig.show()
        
    if return_lists:
        return time_method, number_features, number_instances, list_did, number_classes, diff_with_complete

def plot_cumulative_importance_methods(dict_results, model_name, method_names, save_path=None):
    df_stats = pd.DataFrame(columns = ["time", "time per instance", "number of features", "number of instances", "dataset number", "method"])
    dict_importances = {}
    dict_did_AUC = {}
    for method_name in method_names:
        time_method, number_features, number_instances, list_did, number_classes, diff_with_complete = single_method_analysis(dict_results, model_name, method_name, return_lists=True, show=False)
        
        df_stats_method = pd.DataFrame({"time":time_method,
                                        "time per instance":np.array(time_method)/np.array(number_instances), 
                                        "number of features":number_features,
                                        "number of instances":number_instances, 
                                        "dataset number":list_did, 
                                        "number of classes":number_classes,
                                        "diff_with_complete":diff_with_complete})
        df_stats_method["method"] = method_name
        
        df_stats = pd.concat([df_stats,df_stats_method],axis=0)
        
    for method_name in method_names:
        dict_importances[method_name] = {}
        
    for did in dict_results:
        dict_did_AUC[did] = {}
        if model_name in dict_results[did]:
            for method_name in method_names:
                if method_name in dict_results[did][model_name]:
                    importances = (dict_results[did][model_name][method_name]["inf"].abs().mean().sort_values(ascending=False).cumsum()/dict_results[did][model_name][method_name]["inf"].abs().mean().sum()).values
                    dict_did_AUC[did][method_name] = np.trapz(importances, dx=1/len(importances))
                    if len(importances) not in dict_importances[method_name]:
                        dict_importances[method_name][len(importances)] = []
                    dict_importances[method_name][len(importances)].append(importances)
            
    if save_path:
        Path(save_path+"cumulative_importances/").mkdir(parents=True, exist_ok=True)
            
    for n_features in np.sort(df_stats["number of features"].unique()):
        df_importances = pd.DataFrame(columns=list(range(n_features+1)))
        
        dict_AUC = {}
        for method_name in method_names:
            df_importances_method = pd.DataFrame(dict_importances[method_name][n_features],columns=list(range(1,n_features+1)))
            df_importances_method[0] = 0.0
            df_importances_method["method"] = method_name
            
            df_importances = pd.concat([df_importances,df_importances_method], ignore_index=True)
            dict_AUC[method_name] = method_name+" - AUC = %.3f" % np.trapz(df_importances.groupby("method").mean().loc[method_name], dx=1/n_features)

        df_importances_melt = df_importances.melt(id_vars=["method"])
        df_importances_melt["method"] = df_importances_melt["method"].replace(dict_AUC)
        
        plt.figure(figsize=(10,7))
        sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="method",ci=None, palette=colormap[:len(method_names)])
        plt.xlabel("Number of most-important features",fontsize=16)
        plt.xticks(ticks=[i for i in range(n_features+1)], labels=[i for i in range(n_features+1)],fontsize=14)
        plt.ylabel("Importance proportion",fontsize=16)
        plt.yticks(fontsize=14)
        plt.title("Cumulative importance proportion of most-important variables by method\nResults over {} datasets".format(df_importances.groupby("method").count().iloc[0,0]),fontsize=12)
        
        y_texts = []
        df_importances_melt["MeanAUC"] = df_importances_melt["method"].str[-5:].astype(float)
        for i,method in enumerate(df_importances_melt.groupby("method").mean().sort_values("MeanAUC").index.str[:-14]):
            x_base = (i+1)%n_features  #(n_features//2 - (len(method_names)//2) + i - 1)%n_features
            y_model = df_importances_melt.loc[(df_importances_melt["method"].str.startswith(method+' ')) & (df_importances_melt["variable"]==x_base),"value"].mean()
            y_text = y_model
            while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                y_text += 0.03
            y_texts.append(y_text)
            plt.annotate(method, xytext=(n_features*0.85, y_text), xy=(x_base, y_model), arrowprops=dict(facecolor='black', arrowstyle="-"))
        if save_path: plt.savefig(save_path+"cumulative_importances/cumulative_importances_"+str(n_features)+".jpg")
        plt.legend(prop={"size":10})
        plt.show()
        
    
def plot_cumulative_importance_models(dict_results, model_names, method_name, save_path=None):
    df_stats = pd.DataFrame(columns = ["time", "time per instance", "number of features", "number of instances", "dataset number", "model_name"])
    dict_importances = {}
    dict_did_AUC = {}
    
    for model_name in model_names:
        time_model = []
        number_features = []
        number_instances = []
        list_did = []
        number_classes = []
        diff_with_complete=[]
        
        dict_importances[model_name] = {}
        for did in dict_results:
            dict_did_AUC[did] = {}
            if model_name in dict_results[did]:
                if method_name in dict_results[did][model_name]:
                    time_model.append(dict_results[did][model_name][method_name]["time"])
                    diff_with_complete.append((dict_results[did][model_name]["complete"]["inf"] - dict_results[did][model_name][method_name]["inf"]).abs().mean().mean())
                    number_features.append(dict_results[did]["X"].shape[1])
                    number_instances.append(dict_results[did]["X"].shape[0])
                    list_did.append(did)
                    try:
                        number_classes.append(dict_results[did]["y"].iloc[:,0].unique().shape[0])
                    except:
                        number_classes.append(dict_results[did]["y"].unique().shape[0])
                    importances = (dict_results[did][model_name][method_name]["inf"].abs().mean().sort_values(ascending=False).cumsum()/dict_results[did][model_name][method_name]["inf"].abs().mean().sum()).values
                    dict_did_AUC[did][method_name] = np.trapz(importances, dx=1/len(importances))
                    if len(importances) not in dict_importances[model_name]:
                        dict_importances[model_name][len(importances)] = []
                    dict_importances[model_name][len(importances)].append(importances)
                    
        df_stats_model = pd.DataFrame({"time":time_model,
                                "time per instance":np.array(time_model)/np.array(number_instances), 
                                "number of features":number_features,
                                "number of instances":number_instances, 
                                "dataset number":list_did, 
                                "number of classes":number_classes,
                                "diff_with_complete":diff_with_complete})
        df_stats_model["model_name"] = model_name
        
        df_stats = pd.concat([df_stats,df_stats_model],axis=0)
                    
    if save_path:
        Path(save_path+"cumulative_importances/").mkdir(parents=True, exist_ok=True)
        
    for n_features in np.sort(df_stats["number of features"].unique()):
        df_importances = pd.DataFrame(columns=list(range(n_features+1)))
        dict_AUC = {}
        for model_name in model_names:
            if n_features in dict_importances[model_name]:
                df_importances_model = pd.DataFrame(dict_importances[model_name][n_features],columns=list(range(1,n_features+1)))
                df_importances_model[0] = 0.0
                df_importances_model["model_name"] = model_name

                df_importances = pd.concat([df_importances,df_importances_model], ignore_index=True)
                dict_AUC[model_name] = model_name+" - AUC = %.3f" % np.trapz(df_importances.groupby("model_name").mean().loc[model_name], dx=1/n_features)
        
        df_importances_melt = df_importances.melt(id_vars=["model_name"])
        df_importances_melt["model_name"] = df_importances_melt["model_name"].replace(dict_AUC)
        plt.figure(figsize=(10,7))
        sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="model_name", ci=None)
        plt.xlabel("Number of most-important features",fontsize=16)
        plt.ylabel("Importance proportion",fontsize=16)
        plt.xticks(ticks=[i for i in range(n_features+1)], labels=[i for i in range(n_features+1)],fontsize=14)
        plt.yticks(fontsize=14)
        plt.title("Cumulative importance proportion of most-important variables by model\nResults over {} datasets for the {} method".format(df_importances.groupby("model_name").count().iloc[0,0], method_name))

        y_texts = []
        df_importances_melt["MeanAUC"] = df_importances_melt["model_name"].str[-5:].astype(float)
        for i,model in enumerate(df_importances_melt.groupby("model_name").mean().sort_values("MeanAUC").index.str[:-14]):
            x_base = (i+1)%n_features  #(n_features//2 - (len(method_names)//2) + i - 1)%n_features
            y_method = df_importances_melt.loc[(df_importances_melt["model_name"].str.startswith(model+' ')) & (df_importances_melt["variable"]==x_base),"value"].mean()
            y_text = y_method
            while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                y_text += 0.03
            y_texts.append(y_text)
            plt.annotate(model, xytext=(n_features*0.8, y_text), xy=(x_base, y_method), arrowprops=dict(facecolor='black', arrowstyle="-"))
        if save_path: plt.savefig(save_path+"cumulative_importances/cumulative_importances_"+str(n_features)+".jpg")
        plt.show()

# This method returns two DataFrames from the dictionary of all results
# These DataFrames are easier to manipulate to plot graphs
def dataframe_from_dict_results(dict_results):
    df_stats = pd.DataFrame(columns = ["time", "time per instance", "diff with complete", "number of features", "number of instances", "dataset number", "model_name", "method_name", "AUC"])
    dict_importances = {}
    #dict_did_AUC = {}
    
    for did in dict_results:
        for model_name in dict_results[did]:
            if model_name not in ["X","y"]:
                if model_name not in dict_importances:
                    dict_importances[model_name]={}
                for method_name in dict_results[did][model_name]:
                    if method_name not in dict_importances[model_name]:
                        dict_importances[model_name][method_name] = {}
                    time = dict_results[did][model_name][method_name]["time"]
                    diff_with_complete = (dict_results[did][model_name]["complete"]["inf"] - dict_results[did][model_name][method_name]["inf"]).abs().mean().mean()
                    number_features = dict_results[did]["X"].shape[1]
                    number_instances = dict_results[did]["X"].shape[0]
                    try:
                        number_classes = dict_results[did]["y"].iloc[:,0].unique().shape[0]
                    except:
                        number_classes = dict_results[did]["y"].unique().shape[0]
                    importances = np.concatenate(([0],(dict_results[did][model_name][method_name]["inf"].abs().mean().sort_values(ascending=False).cumsum()/dict_results[did][model_name][method_name]["inf"].abs().mean().sum()).values))
                    AUC = np.trapz(importances, dx=1/(len(importances)-1))
                    if len(importances)-1 not in dict_importances[model_name][method_name]:
                        dict_importances[model_name][method_name][len(importances)-1] = []
                    dict_importances[model_name][method_name][len(importances)-1].append(importances)

                    df_stats_model = pd.DataFrame({"time":[time],
                                                   "time per instance":[time/number_instances],
                                                   "number of features":[number_features],
                                                   "number of instances":[number_instances], 
                                                   "dataset number":[did], 
                                                   "number of classes":[number_classes],
                                                   "diff with complete":[diff_with_complete],
                                                   "model_name":[model_name],
                                                   "method_name":[method_name],
                                                   "AUC":AUC})

                    df_stats = pd.concat([df_stats,df_stats_model],axis=0)
                    
    return df_stats, dict_importances

def multi_plot_cumulative_importance_models(df_stats=None, dict_importances=None, dict_results=None, save_path=None):
    if (df_stats is None) | (dict_importances is None):
        df_stats, dict_importances = dataframe_from_dict_results(dict_results)
        
    ###################################
    ### PLOT CUMULATIVE IMPORTANCES ###
    ###################################
    
    method_names = np.sort(df_stats["method_name"].unique())
    
    dict_colors = {method_name:colormap[i] for i, method_name in enumerate(np.sort(df_stats["model_name"].unique()))}
    dict_markers = {method_name:[".","s","D","P","*","d"][i] for i, method_name in enumerate(np.sort(df_stats["model_name"].unique()))}
   
    for n_features in np.sort(df_stats["number of features"].unique()):
        fig, axes = plt.subplots(1,method_names.shape[0],figsize=(10*method_names.shape[0],10),sharey=True)
        for i,method_name in enumerate(method_names):
            ax = axes[i]
            plt.sca(ax)
            df_importances = pd.DataFrame(columns=list(range(10+1)))

            dict_AUC = {}
            model_names = np.sort(df_stats.loc[df_stats["method_name"]==method_name,"model_name"].unique())
            for model_name in model_names:
                df_importances_model = pd.DataFrame(dict_importances[model_name][method_name][n_features],columns=list(range(0,n_features+1)))
                df_importances_model["model_name"] = model_name
                df_importances_model["method_name"] = method_name

                df_importances = pd.concat([df_importances,df_importances_model], ignore_index=True)

            df_importances_melt = df_importances.melt(id_vars=["model_name","method_name"])

            sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="model_name",ci=None, palette=dict_colors,ax=ax)
            if n_features == np.sort(df_stats["number of features"].unique())[-1]: 
                plt.xlabel("Number of most-important features")#,fontsize=16)
            else:
                plt.xlabel(None)
            plt.xticks(ticks=[i for i in range(n_features+1)], labels=[i for i in range(n_features+1)])
            plt.xlim(right=n_features+2*(n_features/10.))
            plt.ylabel("Cumulative importance proportion")
            if n_features == np.sort(df_stats["number of features"].unique())[0]: plt.title(method_name)

            y_texts = []
            for i,model_name in enumerate(df_stats.loc[(df_stats["number of features"]==n_features) & (df_stats["method_name"]==method_name)].groupby("model_name")["AUC"].mean().sort_values().index):
                x_base = (i+1)%n_features  #(n_features//2 - (len(model_names)//2) + i - 1)%n_features
                y_method = df_importances_melt.loc[(df_importances_melt["model_name"]==model_name) & (df_importances_melt["variable"]==x_base),"value"].mean()
                y_text = y_method
                while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                    y_text += 0.03
                y_texts.append(y_text)
                plt.annotate(model_name, xytext=(n_features*0.8, y_text), xy=(x_base, y_method), arrowprops=dict(facecolor='black', arrowstyle="-"), fontsize=18)
        handles_tot = []
        labels_tot = []
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
        by_label = dict(zip(labels_tot, handles_tot))
        [c.get_legend().remove() for c in axes]
        #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
        if n_features == np.sort(df_stats["number of features"].unique())[-1]: fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
        plt.tight_layout()
        #if save_path: plt.savefig(save_path+"multi_models_cumulative_importances.jpg", bbox_inches="tight")
        plt.show()
        
def multi_plot_cumulative_importance_methods(df_stats=None, dict_importances=None, dict_results=None, save_path=None):
    if (df_stats is None) | (dict_importances is None):
        df_stats, dict_importances = dataframe_from_dict_results(dict_results)
        
    ###################################
    ### PLOT CUMULATIVE IMPORTANCES ###
    ###################################
    
    model_names = np.sort(df_stats["model_name"].unique())
    
    dict_colors = {model_name:colormap[i] for i, model_name in enumerate(np.sort(df_stats["method_name"].unique()))}
    dict_markers = {model_name:[".","s","D","P","*","d"][i] for i, model_name in enumerate(np.sort(df_stats["method_name"].unique()))}
    
    
    for n_features in np.sort(df_stats["number of features"].unique()):
        fig, axes = plt.subplots(1,model_names.shape[0],figsize=(10*model_names.shape[0],10),sharey=True)
        for i,model_name in enumerate(model_names):
            ax = axes[i]
            plt.sca(ax)
            df_importances = pd.DataFrame(columns=list(range(n_features+1)))

            dict_AUC = {}
            method_names = np.sort(df_stats.loc[df_stats["model_name"]==model_name,"method_name"].unique())
            for method_name in method_names:
                df_importances_method = pd.DataFrame(dict_importances[model_name][method_name][n_features],columns=list(range(0,n_features+1)))
                #df_importances_method[0] = 0.0
                df_importances_method["method_name"] = method_name
                df_importances_method["model_name"] = model_name

                df_importances = pd.concat([df_importances,df_importances_method], ignore_index=True)
                #dict_AUC[method_name] = method_name+" - AUC = %.3f" % np.trapz(df_importances.groupby("method_name").mean().loc[method_name], dx=1/n_features)

            df_importances_melt = df_importances.melt(id_vars=["method_name","model_name"])
            #df_importances_melt["method"] = df_importances_melt["method"].replace(dict_AUC)

            sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="method_name",ci=None, palette=dict_colors,ax=ax)
            if n_features == np.sort(df_stats["number of features"].unique())[-1]: 
                plt.xlabel("Number of most-important features")#,fontsize=16)
            else:
                plt.xlabel(None)
            plt.xticks(ticks=[i for i in range(n_features+1)], labels=[i for i in range(n_features+1)])#,fontsize=14)
            plt.xlim(right=n_features+2*(n_features/10.))
            plt.ylabel("Cumulative importance proportion")
            if n_features == np.sort(df_stats["number of features"].unique())[0]: plt.title(model_name)#,fontsize=14)

            y_texts = []
            #df_importances_melt["MeanAUC"] = df_importances_melt["method_name"].str[-5:].astype(float)
            for i,method_name in enumerate(df_stats.loc[(df_stats["number of features"]==n_features) & (df_stats["model_name"]==model_name)].groupby("method_name")["AUC"].mean().sort_values().index):
                x_base = (i+1)%n_features  #(n_features//2 - (len(method_names)//2) + i - 1)%n_features
                y_model = df_importances_melt.loc[(df_importances_melt["method_name"]==method_name) & (df_importances_melt["variable"]==x_base),"value"].mean()
                y_text = y_model
                while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                    y_text += 0.03
                y_texts.append(y_text)
                plt.annotate(method_name, xytext=(n_features*0.9, y_text), xy=(x_base, y_model), arrowprops=dict(facecolor='black', arrowstyle="-"), fontsize=18)
            #if save_path: plt.savefig(save_path+"cumulative_importances/cumulative_importances_"+str(n_features)+".jpg")
        handles, labels = ax.get_legend_handles_labels()
        for handle in handles:
            handle.set_linewidth(3)
        [c.get_legend().remove() for c in axes]
        #axes[0][0].legend(handles, labels, bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
        if n_features == np.sort(df_stats["number of features"].unique())[-1]: fig.legend(handles, labels, bbox_to_anchor=(0.5,0), loc='upper center')
        plt.tight_layout()
        #if save_path: plt.savefig(save_path+"multi_methods_cumulative_importances.jpg", bbox_inches='tight')
        plt.show()
    

# This method generates and plots the graphs of Section 4.2
def multi_plots_methods(df_stats=None, dict_importances=None, dict_results=None, save_path=None):
    if (df_stats is None) | (dict_importances is None):
        df_stats, dict_importances = dataframe_from_dict_results(dict_results)
        
    ###################################
    ### PLOT CUMULATIVE IMPORTANCES ###
    ###################################
    
    model_names = np.sort(df_stats["model_name"].unique())
    fig, axes = plt.subplots(2,2,figsize=(20,15),sharex=True,sharey=True)
    
    dict_colors = {model_name:colormap[i] for i, model_name in enumerate(np.sort(df_stats["method_name"].unique()))}
    dict_markers = {model_name:[".","s","D","P","*","d"][i] for i, model_name in enumerate(np.sort(df_stats["method_name"].unique()))}
    
    for i,model_name in enumerate(model_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        df_importances = pd.DataFrame(columns=list(range(10+1)))
        
        dict_AUC = {}
        method_names = np.sort(df_stats.loc[df_stats["model_name"]==model_name,"method_name"].unique())
        for method_name in method_names:
            df_importances_method = pd.DataFrame(dict_importances[model_name][method_name][10],columns=list(range(0,10+1)))
            #df_importances_method[0] = 0.0
            df_importances_method["method_name"] = method_name
            df_importances_method["model_name"] = model_name
            
            df_importances = pd.concat([df_importances,df_importances_method], ignore_index=True)
            #dict_AUC[method_name] = method_name+" - AUC = %.3f" % np.trapz(df_importances.groupby("method_name").mean().loc[method_name], dx=1/n_features)

        df_importances_melt = df_importances.melt(id_vars=["method_name","model_name"])
        #df_importances_melt["method"] = df_importances_melt["method"].replace(dict_AUC)
        
        sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="method_name",ci=None, palette=dict_colors,ax=ax)
        plt.xlabel("Number of most-important features")#,fontsize=16)
        plt.xticks(ticks=[i for i in range(10+1)], labels=[i for i in range(10+1)])#,fontsize=14)
        plt.xlim(right=12)
        plt.ylabel("Cumulative importance proportion")#,fontsize=16)
        #plt.yticks(fontsize=14)
        plt.title(model_name)#,fontsize=14)
        
        y_texts = []
        #df_importances_melt["MeanAUC"] = df_importances_melt["method_name"].str[-5:].astype(float)
        for i,method_name in enumerate(df_stats.loc[(df_stats["number of features"]==10) & (df_stats["model_name"]==model_name)].groupby("method_name")["AUC"].mean().sort_values().index):
            x_base = (i+1)%10  #(n_features//2 - (len(method_names)//2) + i - 1)%n_features
            y_model = df_importances_melt.loc[(df_importances_melt["method_name"]==method_name) & (df_importances_melt["variable"]==x_base),"value"].mean()
            y_text = y_model
            while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                y_text += 0.03
            y_texts.append(y_text)
            plt.annotate(method_name, xytext=(10*0.9, y_text), xy=(x_base, y_model), arrowprops=dict(facecolor='black', arrowstyle="-"), fontsize=18)
        #if save_path: plt.savefig(save_path+"cumulative_importances/cumulative_importances_"+str(n_features)+".jpg")
    handles, labels = ax.get_legend_handles_labels()
    for handle in handles:
        handle.set_linewidth(3)
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(handles, labels, bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(handles, labels, bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_methods_cumulative_importances.jpg", bbox_inches='tight')
    plt.show()
            
    ################
    ### PLOT AUC ###
    ################
    
    fig, axes = plt.subplots(2,2,figsize=(20,15),sharex=True,sharey=True)
    for i,model_name in enumerate(model_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        df_to_plot = df_stats[df_stats["model_name"]==model_name].groupby(["number of features","method_name"])["AUC"].mean().unstack("method_name")
        df_error = (df_stats[df_stats["model_name"]==model_name].groupby(["number of features","method_name"])["AUC"].std() / np.sqrt(df_stats[df_stats["model_name"]==model_name].groupby(["number of features","method_name"])["AUC"].count())).unstack("method_name")
        method_names = df_to_plot.columns
        for method_name in method_names:
            plt.errorbar(df_to_plot.index,
                         df_to_plot[method_name],
                         yerr=df_error[method_name],
                         color=dict_colors[method_name], capsize=2, label=method_name, marker=dict_markers[method_name])
        plt.title(model_name)
        if ((i//2)+1 == len(model_names)//2):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("AUC")
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_methods_AUC.jpg", bbox_inches='tight')
    plt.show()
            
    ##############################
    ### PLOT TIME PER INSTANCE ###
    ##############################
    
    df_stats_grouped = df_stats.groupby(["model_name","method_name","number of features"],as_index=False)[["time per instance","diff with complete"]].mean()
    df_stats_grouped["std_time"] = df_stats.groupby(["model_name","method_name","number of features"],as_index=False).std()["time per instance"]
    df_stats_grouped["std_error_time"] = df_stats.groupby(["model_name","method_name","number of features"],as_index=False).std()["time per instance"]/np.sqrt(df_stats.groupby(["model_name","method_name","number of features"],as_index=False).count()["time per instance"])
    df_stats_grouped["std_diff"] = df_stats.groupby(["model_name","method_name","number of features"],as_index=False).std()["diff with complete"]
    df_stats_grouped["std_error_diff"] = df_stats.groupby(["model_name","method_name","number of features"],as_index=False).std()["diff with complete"]/np.sqrt(df_stats.groupby(["model_name","method_name","number of features"],as_index=False).count()["diff with complete"])
    
    fig, axes = plt.subplots(2,2,figsize=(20,15),sharex=True,sharey=True)
    for i,model_name in enumerate(model_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        method_names = np.sort(df_stats_grouped.loc[df_stats_grouped["model_name"]==model_name,"method_name"].unique())
        for method_name in method_names:
            plt.errorbar(df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"number of features"],
                         df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"time per instance"],
                         yerr=df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"std_error_time"],
                         color=dict_colors[method_name], capsize=2, label=method_name, marker=dict_markers[method_name])
        plt.yscale("log")
        plt.title(model_name)
        if ((i//2)+1 == len(model_names)//2):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("Time per instance (seconds)")
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_methods_time_per_instance.jpg", bbox_inches='tight')
    plt.show()
       
    ###############################
    ### PLOT DIFF WITH COMPLETE ###
    ###############################
    
    fig, axes = plt.subplots(2,2,figsize=(20,15),sharex=True,sharey=True)
    for i,model_name in enumerate(model_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        method_names = np.sort(df_stats_grouped.loc[df_stats_grouped["model_name"]==model_name,"method_name"].unique())
        for method_name in method_names[method_names!="complete"]:
            if not ((model_name=="XGBoost") & (method_name=="treeSHAPapprox")):
                plt.errorbar(df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"number of features"],
                             df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"diff with complete"],
                             yerr=df_stats_grouped.loc[(df_stats_grouped["model_name"]==model_name) & (df_stats_grouped["method_name"]==method_name),"std_error_diff"],
                             color=dict_colors[method_name], capsize=2, label=method_name, marker=dict_markers[method_name])
        plt.title(model_name)
        if ((i//2)+1 == len(model_names)//2):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("Mean absolute difference with\ncomplete method")
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_methods_diff_complete.jpg", bbox_inches='tight')
    plt.show()
    
    
# This method generates and plots the graphs of Section 4.3
def multi_plots_models(df_stats=None, dict_importances=None, dict_results=None, save_path=None):
    if (df_stats is None) | (dict_importances is None):
        df_stats, dict_importances = dataframe_from_dict_results(dict_results)
        
    ###################################
    ### PLOT CUMULATIVE IMPORTANCES ###
    ###################################
    
    method_names = np.sort(df_stats["method_name"].unique())
    fig, axes = plt.subplots((len(method_names)+1)//2,2,figsize=(20,8*((len(method_names)+1)//2)),sharex=True,sharey=True)
    
    dict_colors = {method_name:colormap[i] for i, method_name in enumerate(np.sort(df_stats["model_name"].unique()))}
    dict_markers = {method_name:[".","s","D","P","*","d"][i] for i, method_name in enumerate(np.sort(df_stats["model_name"].unique()))}
    
    for i,method_name in enumerate(method_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        df_importances = pd.DataFrame(columns=list(range(10+1)))
        
        dict_AUC = {}
        model_names = np.sort(df_stats.loc[df_stats["method_name"]==method_name,"model_name"].unique())
        for model_name in model_names:
            df_importances_model = pd.DataFrame(dict_importances[model_name][method_name][10],columns=list(range(0,10+1)))
            df_importances_model["model_name"] = model_name
            df_importances_model["method_name"] = method_name
            
            df_importances = pd.concat([df_importances,df_importances_model], ignore_index=True)

        df_importances_melt = df_importances.melt(id_vars=["model_name","method_name"])
        
        sns.lineplot(data=df_importances_melt, x="variable", y="value", hue="model_name",ci=None, palette=dict_colors,ax=ax)
        plt.xlabel("Number of most-important features")#,fontsize=16)
        plt.xticks(ticks=[i for i in range(10+1)], labels=[i for i in range(10+1)])#,fontsize=14)
        plt.xlim(right=11.5)
        plt.ylabel("Cumulative importance proportion")#,fontsize=16)
        #plt.yticks(fontsize=14)
        plt.title(method_name)#,fontsize=14)
        
        y_texts = []
        for i,model_name in enumerate(df_stats.loc[(df_stats["number of features"]==10) & (df_stats["method_name"]==method_name)].groupby("model_name")["AUC"].mean().sort_values().index):
            x_base = (i+1)%10  #(n_features//2 - (len(model_names)//2) + i - 1)%n_features
            y_method = df_importances_melt.loc[(df_importances_melt["model_name"]==model_name) & (df_importances_melt["variable"]==x_base),"value"].mean()
            y_text = y_method
            while np.any((y_text-0.03 < np.array(y_texts)) & (y_text+0.03 > np.array(y_texts))):
                y_text += 0.03
            y_texts.append(y_text)
            plt.annotate(model_name, xytext=(10*0.8, y_text), xy=(x_base, y_method), arrowprops=dict(facecolor='black', arrowstyle="-"), fontsize=18)
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_models_cumulative_importances.jpg", bbox_inches="tight")
    plt.show()

    ################
    ### PLOT AUC ###
    ################
    
    fig, axes = plt.subplots((len(method_names)+1)//2,2,figsize=(20,8*((len(method_names)+1)//2)),sharex=True,sharey=True)
    for i,method_name in enumerate(method_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        df_to_plot = df_stats[df_stats["method_name"]==method_name].groupby(["number of features","model_name"])["AUC"].mean().unstack("model_name")
        df_error = (df_stats[df_stats["method_name"]==method_name].groupby(["number of features","model_name"])["AUC"].std() / np.sqrt(df_stats[df_stats["method_name"]==method_name].groupby(["number of features","model_name"])["AUC"].count())).unstack("model_name")
        model_names = df_to_plot.columns
        for model_name in model_names:
            plt.errorbar(df_to_plot.index,
                         df_to_plot[model_name],
                         yerr=df_error[model_name],
                         color=dict_colors[model_name], capsize=2, label=model_name, marker=dict_markers[model_name])
        plt.title(method_name)
        if ((i//2)+1 == len(method_names)//2):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("AUC")
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    #axes[0][0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0,1), loc='upper left')#,fontsize=14)
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_models_AUC.jpg", bbox_inches="tight")
    plt.show()
    
    
    ##############################
    ### PLOT TIME PER INSTANCE ###
    ##############################
    
    df_stats_grouped = df_stats.groupby(["method_name","model_name","number of features"],as_index=False)[["time per instance","diff with complete"]].mean()
    df_stats_grouped["std_time"] = df_stats.groupby(["method_name","model_name","number of features"],as_index=False).std()["time per instance"]
    df_stats_grouped["std_error_time"] = df_stats.groupby(["method_name","model_name","number of features"],as_index=False).std()["time per instance"]/np.sqrt(df_stats.groupby(["method_name","model_name","number of features"],as_index=False).count()["time per instance"])
    df_stats_grouped["std_diff"] = df_stats.groupby(["method_name","model_name","number of features"],as_index=False).std()["diff with complete"]
    df_stats_grouped["std_error_diff"] = df_stats.groupby(["method_name","model_name","number of features"],as_index=False).std()["diff with complete"]/np.sqrt(df_stats.groupby(["method_name","model_name","number of features"],as_index=False).count()["diff with complete"])
    
    fig, axes = plt.subplots((len(method_names)+1)//2,2,figsize=(20,8*((len(method_names)+1)//2)),sharex=True,sharey=True)
    for i,method_name in enumerate(method_names):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        model_names = np.sort(df_stats_grouped.loc[df_stats_grouped["method_name"]==method_name,"model_name"].unique())
        for model_name in model_names:
            plt.errorbar(df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"number of features"],
                         df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"time per instance"],
                         yerr=df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"std_error_time"],
                         color=dict_colors[model_name], capsize=2, label=model_name, marker=dict_markers[model_name])
        plt.yscale("log")
        plt.title(method_name)
        if ((i//2)+1 == len(method_names)//2):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("Time per instance (seconds)")
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
    by_label = dict(zip(labels_tot, handles_tot))
    [[c.get_legend().remove() for c in r] for r in axes]
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_models_time_per_instance.jpg", bbox_inches="tight")
    plt.show()
       
    ###############################
    ### PLOT DIFF WITH COMPLETE ###
    ###############################
    
    method_names_plot = method_names[(method_names!="complete") & (method_names != "treeSHAPapprox")]
    fig, axes = plt.subplots((len(method_names_plot)+1)//2,2,figsize=(20,8*((len(method_names_plot)+1)//2)),sharex=True,sharey=True)
    for i,method_name in enumerate(method_names_plot):
        ax = axes[i//2][i%2]
        plt.sca(ax)
        model_names = np.sort(df_stats_grouped.loc[df_stats_grouped["method_name"]==method_name,"model_name"].unique())
        for model_name in model_names:
            if not ((model_name=="XGBoost") & (method_name=="treeSHAPapprox")):
                plt.errorbar(df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"number of features"],
                             df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"diff with complete"],
                             yerr=df_stats_grouped.loc[(df_stats_grouped["method_name"]==method_name) & (df_stats_grouped["model_name"]==model_name),"std_error_diff"],
                             color=dict_colors[model_name], capsize=2, label=model_name, marker=dict_markers[model_name])
        plt.title(method_name)
        if ((i//2)+1)==len(axes):
            plt.xlabel("Number of features")
        if (i%2 == 0):
            plt.ylabel("Mean absolute difference with\ncomplete method")
        if ((len(method_names_plot)%2 != 0) & ((i//2)+1==(len(method_names)//2)-1)) & (i%2 != 0):
            print("coucou2")
            plt.xlabel("Number of features")
            ax.tick_params(axis="x", labelbottom=True)
        plt.legend()
    handles_tot = []
    labels_tot = []
    for ax0 in axes:
        for ax in ax0:
            handles, labels = ax.get_legend_handles_labels()
            handles_tot += handles
            labels_tot += labels
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    by_label = dict(zip(labels_tot, handles_tot))
    fig.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5,0), loc='upper center')
    
    if len(method_names_plot)%2 != 0:
        print("coucou")
        for l in axes[len(method_names[method_names!="complete"])//2-1,1].get_xticklabels():
            l.set_visible(True)
        fig.delaxes(axes[-1, 1])
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path+"multi_models_diff_complete.jpg", bbox_inches='tight')
    plt.show()
    
#################################
### Dataset specific analysis ###
#################################

def dataset_test_bootstrap(n_bootstrap, X, y, models, test_size=0.2, problem_type="Classification", mode="classification", fvoid=None, look_at=1, rate=0.25, n_background_samples=10, silent=True, progression_bar=False):  
    dict_bootstrap = {}
    for i in range(n_bootstrap):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y,test_size=test_size)
        dict_bootstrap[i]={"i_train":X_train.index,"i_test":X_test.index}
        for model_name in models:
            dict_bootstrap[i][model_name]={}
            models[model_name].fit(X_train, y_train)
            dict_bootstrap[i][model_name]["train_accuracy"] = sklearn.metrics.accuracy_score(y_train, models[model_name].predict(X_train))
            dict_bootstrap[i][model_name]["test_accuracy"] = sklearn.metrics.accuracy_score(y_test, models[model_name].predict(X_test))

            test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_complete, method_name="complete", y=y, problem_type=problem_type, fvoid=fvoid, look_at=look_at, progression_bar=progression_bar) #Complete
            test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_spearman, method_name="spearman"+str(rate), y=y, rate=rate, problem_type=problem_type, complexity=True, fvoid=fvoid, look_at=look_at, progression_bar=progression_bar) #Spearman
            test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_lime, method_name="LIME", mode=mode, look_at=look_at, num_samples=100, silent=silent) #LIME
            try:
                test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_kernelSHAP, method_name="kernelSHAP"+(str(n_background_samples) if n_background_samples else ""), look_at=look_at,n_background_samples=n_background_samples, silent=silent) #SHAP
                test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_treeSHAP_approx, method_name="treeSHAPapprox", look_at=look_at) #SHAP
                test_and_store(dict_bootstrap, i, X, models[model_name], model_name, explanation_funct=explanation_values_treeSHAP, method_name="treeSHAP", look_at=look_at) #SHAP
            except:
                print("Model not suited for TreeSHAP or kernelSHAP")
                
    dict_bootstrap["Mean"] = {}
    for model_name in models:
        dict_bootstrap["Mean"][model_name] = {}
        dict_bootstrap["Mean"][model_name]["train_accuracy"] = np.mean([dict_bootstrap[i][model_name]["train_accuracy"] for i in range(n_bootstrap)])
        dict_bootstrap["Mean"][model_name]["test_accuracy"] = np.mean([dict_bootstrap[i][model_name]["test_accuracy"] for i in range(n_bootstrap)])
        
        for method_name in dict_bootstrap[0][model_name]:
            if not method_name.endswith("accuracy"):
                dict_bootstrap["Mean"][model_name][method_name] = {}
                dict_bootstrap["Mean"][model_name][method_name]["inf"] = np.mean([dict_bootstrap[i][model_name][method_name]["inf"] for i in range(n_bootstrap)],axis=0)
                dict_bootstrap["Mean"][model_name][method_name]["time"] = np.mean([dict_bootstrap[i][model_name][method_name]["time"] for i in range(n_bootstrap)])
            
    return dict_bootstrap