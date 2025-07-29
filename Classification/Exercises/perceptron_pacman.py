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


# Perceptron implementation for imitation learning
import Helpers.util
from Exercises.perceptron_multiclass import PerceptronClassifier
from random import uniform
from pacman import GameState


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.epochs = maxIterations
        self.weights = [] 
        

    def convert_data(self, data):
        # fix datatype issues
        # if it comes in inside of a list, pull it out of the list
        if isinstance(data, list):
            data = data[0]

        #data comes as a tuple
        all_moves_features = data[0] #grab the features (dict of action->features)
        legal_moves = data[1]        #grab the list of legal moves from this state
        return_features = {}
        #loop each action
        for key in all_moves_features:
            #convert feature values from dict to list
            all_features = ['foodCount', 'STOP', 'nearest_ghost', 'ghost-0', 'capsule-0', 'food-0', 'food-1', \
                            'food-2', 'food-3', 'food-4', 'capsule count', 'win', 'lose', 'score']
            dict_features = all_moves_features[key] 
            list_features = []
            # grab all feature values & put them in a list
            for feat in all_features:
                if feat not in dict_features:
                    list_features.append(0)
                else:
                    list_features.append(dict_features[feat])
            return_features[key] = list_features

        return (return_features, legal_moves) 


    def classify(self, data):
        #leave this call to convert_data here!
        features, legal_moves = self.convert_data(data)
        
        # get dictionary of weighted sums for all legal_moves
        weighted_sums = {}
        weighted_sum = 0
        for move in legal_moves:
            weighted_sum = 0
            for feature_index, feature in enumerate(features[move]):
                weighted_sum += feature * self.weights[feature_index]
            weighted_sums[move] = weighted_sum


        # get highest key by value
        highest_key = legal_moves[0]
        highest_value = weighted_sums[legal_moves[0]]
        for key, val in weighted_sums.items():
            if val > highest_value:
                highest_key = key
                highest_value = val

        
        #your predicted_label needs to be returned inside of a list for the PacMan game
        predicted_label = highest_key
        return [predicted_label]


    def train(self, train_data, labels):
        data, legal_moves = self.convert_data(train_data[0])

        self.weights = [uniform(-1, 1) for _ in range(0, len(data["Stop"]))]

        for epoch in range(self.epochs):
            correct_predicted = 0

            for obj_index, obj in enumerate(train_data):
                predicted = self.classify(obj)[0]
                print(predicted)
                print(self.classify(obj))
                actual = labels[obj_index]

                if predicted == actual:
                    correct_predicted += 1
                    continue
            
                data, legal_moves = self.convert_data(obj)

                print(data)

                for feature_index, feature in enumerate(data[predicted]):
                    self.weights[feature_index] -= feature
                for feature_index, feature in enumerate(data[actual]):
                    self.weights[feature_index] += feature

            accuracy = (correct_predicted / len(train_data)) * 100
            print('epoch: {} ... accuracy: {}'.format(epoch, accuracy))


