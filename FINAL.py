# Neel Nadkarni, Nathan Stalley, Jehan Balading, Cesar Herrera
# Final error:
# Percent wrong = 23.7%
# Percent correct = 76.3%

import csv
from neural import *
from sklearn.model_selection import train_test_split



buying_map = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
maint_map = {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1}
doors_map = {'2': 1, '3': 2, '4': 3, '5more': 4}
persons_map = {'2': 1, '4': 2, 'more': 3}
lug_boot_map = {'small': 1, 'med': 2, 'big': 3}
safety_map = {'low': 1, 'med': 2, 'high': 3}

def transform_data(data):
    transformed_data = []
    for entry in data:
        mapped_entry = []
        for i in range(len(entry) - 1):
            if i == 0:
                mapped_entry.append(buying_map[entry[i]])
            elif i == 1:
                mapped_entry.append(maint_map[entry[i]])
            elif i == 2:
                mapped_entry.append(doors_map[entry[i]])
            elif i == 3:
                mapped_entry.append(persons_map[entry[i]])
            elif i == 4:
                mapped_entry.append(lug_boot_map[entry[i]])
            elif i == 5:
                mapped_entry.append(safety_map[entry[i]])
        dat = (mapped_entry, [entry[-1]])
        transformed_data.append(dat)
    return transformed_data

def read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        data = list(reader)
    return data

file_path = "car_data.csv"

data = read_csv(file_path)

transformed_data = transform_data(data)

for i in transformed_data:
    if i[1][0] == "unacc":
        i[1][0] = 0
    elif i[1][0] == "acc":
        i[1][0] = 0.3
    elif i[1][0] == "good":
        i[1][0] = 0.6
    elif i[1][0] == "vgood":
        i[1][0] = 1

def closest_value(n):
    # Find the fractional part of the input
    fraction = n % 1
    
    # Define the possible values
    values = [0, 0.3, 0.6, 1]
    
    # Find the closest value
    closest = min(values, key=lambda x: abs(x - fraction))
    
    return closest

car_nn = NeuralNet(6, 1, 1)

xtrain, xtest = train_test_split(transformed_data, test_size=.2)

car_nn.train(xtrain)

error_bound = car_nn.test_with_expected(xtest)

print(error_bound[0])

for i in error_bound:
    i[2][0] = closest_value(i[2][0])

correct_num = 0
incorrect_num = 0

for i in error_bound:
    if i[2][0] == i[1][0]:
        correct_num += 1
    else:
        incorrect_num += 1

denom = correct_num + incorrect_num

print(f"Percent wrong = {round(((incorrect_num / denom)*100), 1)}%")
print(f"Percent correct = {round(((correct_num / denom)*100), 1)}%")
