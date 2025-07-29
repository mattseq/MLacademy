# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# ----------


# Perceptron implementation
import Helpers.util
from random import uniform


class PerceptronClassifier:

    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = None


    def classify(self, data):
        weighted_sums = []
        for i in range(0, 10):
            output_arr =[]
            for feature_index, feature in enumerate(data):
                output_arr.append(feature*self.weights[i][feature_index])

            weighted_sum = sum(output_arr)
            weighted_sums.append(weighted_sum)
        highest_value = max(weighted_sums)
        highest_index = weighted_sums.index(highest_value)

        return highest_index+1
 

    def train(self, train_data, labels):
        for i, obj in enumerate(train_data):
            train_data[i].append(1.0)

        self.weights = [[uniform(-1, 1) for i in range(0, len(train_data[0]))] for j in range(0, 10)]

        for epoch in range(0, self.epochs):
            correct_predicted = 0

            for obj_index, obj in enumerate(train_data):
                predicted = self.classify(obj)
                actual = labels[obj_index]
                error = float(labels[obj_index] - predicted)
                if error == 0:
                    correct_predicted += 1
                    continue

                for feature_index, feature in enumerate(obj):
                    self.weights[predicted-1][feature_index] -= feature
                    self.weights[actual-1][feature_index] += feature
            
            accuracy = (correct_predicted / len(train_data)) * 100

            print('epoch: {} ... accuracy: {}%'.format(epoch, accuracy))

