"""
This python file stores functions for basic xai functions such as
grabbing a sample.
"""
import pandas as pd
import numpy as np

"""
Grabs a single sample of the data
"""
def grab_sample(X_test, y_test, sample_number):

    # grabs a sample
    sample = X_test.iloc[sample_number]
    sample_y = y_test.iloc[sample_number]
    sample_y = 1 if sample_y == True else 0
    sample = sample.values.reshape(1, -1)

    # converts sample to a single list
    sample_size = np.size(sample[0])

    sample_list = sample[0]
    return sample, sample_list