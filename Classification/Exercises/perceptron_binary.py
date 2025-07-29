import enum
from random import uniform


# Make a prediction with weights
def classify(row, weights):
    output_arr =[]
    for i, feature in enumerate(row):
        output_arr.append(feature*weights[i])

    weighted_sum = sum(output_arr)
    # print(weighted_sum)
    if weighted_sum >= 0:
        # print(1)
        return 1
    else:
        return 0
 
#Estimate Perceptron weights using stochastic gradient descent
def train(train_data, n_epoch, l_rate=1):

    # get labels from train_data and remove them from train_data, also add bias to train_data features
    labels = []
    for i, obj in enumerate(train_data):
        labels.append(obj[-1])
        # print(obj[-1])
        # del obj[-1]
        train_data[i][-1] = 1.0
    
    print(len(train_data[0]))

    # initialize weights randomly
    weights = [uniform(-1, 1) for _ in range(0, len(train_data[0]))]


    for epoch in range(0, n_epoch):
        correct_predicted = 0

        for obj_index, obj in enumerate(train_data):
            predicted = classify(obj, weights)
            error = float(labels[obj_index] - predicted)
            if error == 0:
                correct_predicted += 1
                continue

            for feature_index, feature in enumerate(obj):
                weights[feature_index] = weights[feature_index] + l_rate*(error * feature)
        
        accuracy = (correct_predicted / len(train_data)) * 100

        print('epoch: {} ... accuracy: {}'.format(epoch, accuracy))