"""
Useful functions that seem to be reused.
"""

import pandas as pd
from colorama import Style, Fore
import xlsxwriter
import os
from openpyxl import load_workbook


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

# marker for location in code
def print_marker():
    print("\n\n Made it Here \n\n")

# print variable and ends programs for ease of visibility
def print_and_end(variable):
    print(variable)
    exit(0)

# print dataset information
def display_dataset_information(dataset_info):

    id = dataset_info["id"]
    name = dataset_info["name"]
    num_features = dataset_info["num_features"]
    num_instances = dataset_info["num_instances"]

    print("\n")
    print(f"Dataset Id: {id}")
    print(f"Dataset Name: {name}")
    print(f"Number of Features: {num_features}")
    print(f"Number of Instances: {num_instances}")
    print("\n")


# format results into a excel sheet
def create_excel_sheet(dataset_info, experiment_results):
    # initialization
    filename = "dataset_results.xlsx"
    id = dataset_info["id"]
    name = dataset_info["name"]
    num_features = dataset_info["num_features"]
    num_instances = dataset_info["num_instances"]

    model_names = ["logistic_regression", "support_vector_machine", "random_forest"]
    base_xai_approach_names = ["lime", "complete", "kernelshap", "spearman", "treeshap", "treeshap_approx"]
    swarm_approach_names = ["pso", "bat", "abc"]

    # check if the file exists
    if not os.path.exists(filename):
        # create excel workbook
        workbook = xlsxwriter.Workbook(filename)

        # add worksheets
        workbook.add_worksheet("dataset")
        for model_name in model_names:
            workbook.add_worksheet(model_name)
        
        workbook.close()

    output_list = []

    # append to excel sheet
    
    
    for model_name in model_names: # loop through models
        # df_existing = pd.read_excel(filename, sheet_name = model_name)
        df_combined = pd.DataFrame()

        for base_xai_approach_name in base_xai_approach_names:
            dict_first_part = {"dataset_id": [id], "method": [base_xai_approach_name]}
            final_dict = {**dict_first_part, **experiment_results[model_name]["base_xai"][base_xai_approach_name]}
            df_new = pd.DataFrame(final_dict)

            df_combined = df_combined._append(df_new, ignore_index = True)

        # df_output = df_existing._append(df_combined, ignore_index = True)
        output_list.append(df_combined)

    print(f"Output List: \n{output_list}\n")

    output_df = pd.concat([output_list[0], output_list[1], output_list[2]])
    output_df.to_excel(filename, sheet_name = "sheet1", index = False)

    # output_list[0].to_excel(filename, sheet_name = "sheet1", index = False)
    # output_list[1].to_excel(filename, sheet_name = "sheet2", index = False)
    # output_list[2].to_excel(filename, sheet_name = "sheet3", index = False)
