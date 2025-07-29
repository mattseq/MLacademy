# Perceptron Algorithm on the Sonar Dataset
import random
from csv import reader
from Exercises.perceptron_binary import *
 

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset
 
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


# Test the students' perceptron on the sonar dataset

# load and prepare data
filename = 'Helpers/sonar_all-data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
    str_column_to_float(dataset, i)
# convert string class to integers
str_column_to_int(dataset, len(dataset[0])-1)
random.seed(10)
random.shuffle(dataset)
print("Dataset loaded", len(dataset), "records.")

# run perceptron training
n_epoch = 100
train(dataset, n_epoch)




