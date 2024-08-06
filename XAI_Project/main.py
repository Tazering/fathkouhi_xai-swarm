"""
This python file is the main driver for the experiment.
A signficant portion of this code was made to be similar to Emmanuel Doumard et al's 
experiment found in:

github.com/EmmanuelDoumard/local_explanation_comparative_study

The largest difference is the implementation of our XAI algorithms that use SWARM 
optimizers as also another approach for explainability.
"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import XAI_Swarm_Opt
import XAI
import time
import pickle

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import make_column_selector as selector
import arff
from colorama import Style, Fore
import utils.data_tools as data_tools



"""
The main function that calls each of the needed steps for the experiment.
"""
def main():

    # testing with iris dataset
    

    

    # preprocess the data
   # X_preprocessed_test, y_preprocessed_test = process_openml_dataset(61, "class") 

    # data_tools.print_variable("X_preprocessed", X_preprocessed)

    # grabs the datasets
    # df_datasets = get_datasets()
    # print(df_datasets.shape)

    return None

main()