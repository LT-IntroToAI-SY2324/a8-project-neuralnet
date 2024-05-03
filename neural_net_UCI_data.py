from typing import Tuple
from neural import *
from sklearn.model_selection import train_test_split
import pandas as pd

def parse_line(line: str, num_categories: int) -> Tuple[List[float], List[float]]:
    """Splits line of CSV into inputs and output (transforming output as appropriate)

    Args:
        line - one line of the CSV as a string
        num_categories - number of categories for one-hot encoding

    Returns:
        tuple of input list and output list
    """
    tokens = line.strip().split(",")  # Strip newline character and split by comma
    output = [float(tokens[0])]  # Assuming the first token is the output

    # One-hot encode categorical features
    input_features = []
    for token in tokens[1:]:
        try:
            input_features.append(float(token))  # If it's a number, directly append
        except ValueError:
            # If it's not a number, it's a categorical feature
            # Apply one-hot encoding
            one_hot_encoded = np.zeros(num_categories)
            category_index = hash(token) % num_categories
            one_hot_encoded[category_index] = 1
            input_features.extend(one_hot_encoded.tolist())

    return input_features, output


def normalize(data: List[Tuple[List[float], List[float]]]):
    """Makes the data range for each input feature from 0 to 1

    Args:
        data - list of (input, output) tuples

    Returns:
        normalized data where input features are mapped to 0-1 range (output already
        mapped in parse_line)
    """
    num_features = len(data[0][0])
    leasts = num_features * [100.0]
    mosts = num_features * [0.0]

    for features, _ in data:
        for j in range(len(features)):
            leasts[j] = min(leasts[j], features[j])
            mosts[j] = max(mosts[j], features[j])

    normalized_data = []
    for features, output in data:
        normalized_features = []
        for j in range(len(features)):
            normalized_feature = (features[j] - leasts[j]) / (mosts[j] - leasts[j])
            normalized_features.append(normalized_feature)
        normalized_data.append((normalized_features, output))

    return normalized_data


with open("car_data.txt", "r") as f:
    training_data = [parse_line(line) for line in f.readlines() if len(line) > 4]

# print(training_data)
td = normalize(training_data)
# print(td)

train, test = train_test_split(td)

nn = NeuralNet(13, 3, 1)
nn.train(train, iters=10000, print_interval=1000, learning_rate=0.2)

for i in nn.test_with_expected(test):
    difference = round(abs(i[1][0] - i[2][0]), 3)
    print(f"desired: {i[1]}, actual: {i[2]} diff: {difference}")