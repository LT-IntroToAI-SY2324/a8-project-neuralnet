# Program for demonstration of one hot encoding

# import libraries
import numpy as np
import pandas as pd

# import the data required
data = pd.read_csv('car_data.txt')
one_hot_encoded_data = pd.get_dummies(data, columns = ['buying', 'maint', 'lug_boot', 'safety', 'class'])