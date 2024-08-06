"""
Useful functions that seem to be reused.
"""

import pandas as pd
from colorama import Style, Fore

# print nonscalar variable
def print_variable(name, variable):
    print(f"===={name}====")
    print(variable)

    try:
        print(f"Shape of {name}: {variable.shape}")
    except:
        print(f"Shape of {name}: {len(variable)}")
    
    print(f"Type of {name}: {type(variable)}\n")

# print scalar variable
def print_generic(name, variable):
    print(f"==== {name} ====")
    print(str(variable) + "\n\n")

# print about a chosen
def print_dataset_sample(dataset_name, sample, sample_y, model):
    print(Style.BRIGHT + Fore.LIGHTCYAN_EX + dataset_name + ":")
    print(Style.BRIGHT + Fore.CYAN + 'X: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample))
    print(Style.BRIGHT + Fore.CYAN + 'y: ', Style.BRIGHT + Fore.LIGHTRED_EX + str(sample_y),'\n')
    print(Style.BRIGHT + Fore.CYAN + 'Blackbox model prediction: ', Style.BRIGHT + Fore.YELLOW + str(model.predict_proba(sample)),'\n')
