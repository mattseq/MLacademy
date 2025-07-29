import random
from Exercises.knn import *
from csv import reader
import sys

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
 
# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

if __name__ == "__main__":
        # Example dataset
    dataset = [
        [1, 2, 0], [2, 3, 0], [3, 4, 1], [5, 6, 1], [8, 8, 1],
        [2, 2, 0], [4, 4, 1], [6, 6, 1], [7, 7, 1], [5, 5, 0]
    ]

    # Test get_distances
    try:
        point = [1, 2]
        data_points = [[2, 3], [3, 4], [4, 5]]
        distances = get_distances(point, data_points)
        print("Distances:", distances)
    except:
        sys.exit("Compilation Error in Distance Calculation")

    # Test normalize_data
    train_data = [[1, 2], [2, 3], [3, 4]]
    test_data = [[2, 2], [4, 5]]
    try:
        normalized_train, normalized_test = normalize_data(train_data, test_data)
        print("Normalized train data:", normalized_train)
        print("Normalized test data:", normalized_test)
    except:
        sys.exit("Compilation Error in Normalization")

    # Test run_knn
    try:
        train_set = [[1, 2, 0], [2, 3, 0], [3, 4, 1], [5, 6, 1]]
        test_set = [[2, 2, 0], [4, 4, 1]]
        k = 3
        y_pred, y_test = run_knn(train_set, test_set, k)
        print("Predicted classes:", y_pred)
        print("True classes:", y_test)
    except:
        sys.exit("Compilation error in run_knn")

    # Test run_CV
    k = 3
    try:
        accuracy = run_CV(dataset, k)
        print("Cross-validated accuracy:", accuracy)
    except:
        sys.exit("Compilation error in Cross Validation")
    
    
    
    
    # Example dataset

    filename = 'Helpers/sonar_all-data.csv'
    dataset = load_csv(filename)
    for i in range(len(dataset[0])-1):
        str_column_to_float(dataset, i)
# convert string class to integers
    str_column_to_int(dataset, len(dataset[0])-1)
    random.shuffle(dataset)
    k = 5
    accuracy = run_CV(dataset, k)
    print("Cross-validated accuracy:", accuracy)