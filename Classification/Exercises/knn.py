from math import sqrt

def normalize_data(train, test):
    mins = train[0][:] # copy to prevent it from making it a reference its stupid ik
    maxes = train[0][:]

    # find mins and maxes for each feature based on train data
    for obj_index, obj in enumerate(train):
        for feature_index, feature in enumerate(obj):
            if feature < mins[feature_index]:
                mins[feature_index] = feature
            if feature > maxes[feature_index]:
                maxes[feature_index] = feature

    # scale training
    for obj_index, obj in enumerate(train):
        for feature_index, feature in enumerate(obj):
            if maxes[feature_index] - mins[feature_index] == 0:
                continue
            train[obj_index][feature_index] = (feature - mins[feature_index]) / (maxes[feature_index] - mins[feature_index])

    # scale testing
    for obj_index, obj in enumerate(test):
        for feature_index, feature in enumerate(obj):
            if maxes[feature_index] - mins[feature_index] == 0:
                continue
            test[obj_index][feature_index] = (feature - mins[feature_index]) / (maxes[feature_index] - mins[feature_index])


    return train, test


def get_distances(point, data):
    distances = []
    for other_point in data:
        distance_each_point = []
        for i, val in enumerate(point):
            distance_each_point.append((val - other_point[i])**2)
        distances.append(sum(distance_each_point)**(1/2))

    return distances
        

def run_knn(train_set, test_set, k):
    # get labels from data
    train_labels = []
    test_labels = []
    for obj_index, obj in enumerate(train_set):
        train_labels.append(obj[-1])
        train_set[obj_index].pop()
    for obj_index, obj in enumerate(test_set):
        test_labels.append(obj[-1])
        test_set[obj_index].pop()
        

    # normalize data
    train, test = normalize_data(train_set, test_set)

    predictions = []

    for point_index, point in enumerate(test_set):
        distances = get_distances(point, train)

        # get list of indexes for k nearest (minimum of distances)
        k_indexes = []
        for current_k in range(k):
            min_index = distances.index(min(distances))
            k_indexes.append(min_index)
            distances[min_index] = max(distances) + 1

        # get most labels
        ones = 0
        zeros = 0
        for k_index in k_indexes:
            if train_labels[k_index] == 1:
                ones += 1
            if train_labels[k_index] == 0:
                zeros += 1

        # add to predictions
        if ones >= zeros:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions, test_labels
            

def run_CV(data, k=3, folds=5):
    pass
