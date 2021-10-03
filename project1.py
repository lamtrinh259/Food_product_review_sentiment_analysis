from string import punctuation, digits
import numpy as np
import random
import utils # can use load_data to read the .tsv data files

# Part I

def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices

def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()

def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    z = np.dot(label,(theta@feature_vector + theta_0))
    if z >= 1: 
        return 0
    else:
        return 1 - z
    raise NotImplementedError


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    loss = []
    for i in range(len(labels)):
        z = np.dot(labels[i],(theta@feature_matrix[i] + theta_0))
        if z >= 1: 
            z = 0
        else:
            z = 1 - z
        loss.append(z)
    return sum(loss) / len(loss)
    raise NotImplementedError

# feature_vector = np.array([[1, 2], [1, 2]])
# label, theta, theta_0 = np.array([1, 1]), np.array([-1, 1]), -0.2
# print(hinge_loss_full(feature_vector, label, theta, theta_0))

def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if np.dot(label, (current_theta @ feature_vector + current_theta_0)) <= 0:
        theta = current_theta + np.dot(label, feature_vector)
        theta_0 = current_theta_0 + label
    else:
        theta = current_theta
        theta_0 = current_theta_0
    return (theta, theta_0)
    raise NotImplementedError

# feature_vector = np.array([1, 2])
# label, theta, theta_0 = 1, np.array([-1, 1]), -1.5
# print(perceptron_single_step_update(feature_vector, label, theta, theta_0))

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # len(matrix[0]) allows to get the number of columns (features) 
    # in the feature matrix, 1D matrix of length n 
    theta = np.zeros([len(feature_matrix[0])])   
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
    return (theta, theta_0)
    raise NotImplementedError

# Test case from the test.py
# feature_matrix = np.array([[1, 2], [-1, 0]])
# labels = np.array([1, 1])
# current_theta = np.zeros([len(feature_matrix[0])])   
# current_theta_0 = 0
# T = 2

#Test case from EdX
# feature_matrix = np.array([[-0.49617046, -0.10213916,  0.4615173,  -0.21847289,
#                             0.26531219, -0.11454821, 0.02457172,  0.38402438,
#                             0.08732508, -0.13354111],
#   [ 0.17748823,  0.34955002, -0.28090295, -0.42587523, -0.23423115,  0.25736214,
#     0.01865076,  0.19919163, -0.14699213, -0.12472268],
#   [ 0.39611082,  0.38613971,  0.49043018, -0.4752383,   0.24606609, -0.47962827,
#   -0.44168412, -0.194519,    0.14511279, -0.14911206],
#   [-0.48155432,  0.17997986, -0.40373332, -0.01263177, -0.15159807, -0.17032547,
#   -0.12805966, -0.49804704,  0.22530995, -0.29756625],
#   [ 0.32486067,  0.23796836, -0.49147082,  0.22525514,  0.31048765, -0.40807908,
#     0.0655446,  -0.19094952, -0.42854669,  0.14811454]])
# labels = np.array([-1,  1,  1, -1, -1])
# T = 5
#Get the number of columns (features) in the feature matrix
# current_theta = np.zeros([len(feature_matrix[0])]) 
# print(perceptron(feature_matrix, labels, T))
# print(perceptron_single_step_update(feature_matrix[0], labels[0], current_theta, current_theta_0))


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta = np.zeros([len(feature_matrix[0])])   
    sum_theta = 0
    avg_theta = 0
    theta_0 = 0
    sum_theta_0 = 0
    avg_theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            sum_theta += theta
            sum_theta_0 += theta_0
    avg_theta = sum_theta / (T*len(labels))
    avg_theta_0 = sum_theta_0 / (T*len(labels))
    return (avg_theta, avg_theta_0)    
    raise NotImplementedError

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # My 1st submission: Already correct on EdX
    # if label * (current_theta@feature_vector.T) + current_theta_0 <= 1:
    # Official correct algorithm
    # if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1
    
    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        theta = (1-eta*L)*current_theta + eta*label*feature_vector
        theta_0 = current_theta_0 + eta*label 
    else: 
        theta = (1-eta*L)*current_theta
        theta_0 = current_theta_0
    return (theta, theta_0)
    raise NotImplementedError

# Test case from EdX
# feature_vector = np.array([[-0.09312743,  0.35804648,  0.08565895, -0.39131367,  0.31911457,
#                   0.10881271, 0.14395856, -0.12581321, 0.16360541, 0.07373427]])
# label = 1
# L = 0.9408870406300012
# eta = 0.3326674633482306
# theta = np.array([[-0.36643546, -0.26709331, 0.21411377, 0.33387355, -0.32386434, 
#         -0.05075449, -0.30664893, -0.32178065,  0.44455982,  0.24358299]])
# theta_0 = 1.7519771454132986
# print(pegasos_single_step_update(feature_vector, label, L, eta, theta, theta_0))
#pegasos_single_step_update output is 
#(['-0.2517402', '-0.1834924', '0.1470956', '0.2293703', '-0.2224940', '-0.0348682',
# '-0.2106670', '-0.2210625', '0.3054115', '0.1673409'], '1.7519771')

def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    # Initialize both theta and theta_0 to 0 
    theta = np.zeros([len(feature_matrix[0])])  # Keep this as a vector
    theta_0 = 0
    helper = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            eta =  1 / np.sqrt(helper)
            theta, theta_0 = pegasos_single_step_update(feature_matrix[i], labels[i], L, eta, theta, theta_0)
            helper += 1
    return (theta, theta_0)  
    raise NotImplementedError

# Test case from test.py
# feature_matrix = np.array([[1, 1], [1, 1]])
# labels = np.array([1, 1])
# T = 1
# L = 1
# print(pegasos(feature_matrix, labels, T, L))

#Test case from EdX
# pegasos input:
# feature_matrix = np.array([[ 0.32453673,  0.06082212,  0.27845097,  0.27124962, -0.48858134],
#   [-0.07490036, -0.2226942,   0.46808161, -0.15484728, -0.06555043],
#   [ 0.48089473,  0.11053774, -0.39253255, -0.45844357,  0.19818921],
#   [ 0.39728286,  0.14426349,  0.23446484, -0.46963688,  0.30978055],
#   [-0.2836313,   0.20048277,  0.10600686, -0.47812081,  0.24772569],
#   [-0.38813183, -0.39082381,  0.02482903,  0.46576666, -0.22720277],
#   [ 0.15482689, -0.16083218,  0.38637948, -0.14209394,  0.05076824],
#   [-0.1238048,  -0.1064888,  -0.28800396, -0.47983335,  0.31652173],
#   [ 0.31485345,  0.30679047, -0.1907081,  -0.0961867,   0.27954887],
#   [ 0.4024408,   0.2990748,   0.34148516, -0.311256,   0.13324454]])
# labels = [-1, -1,  1,  1,  1,  1, -1, -1,  1,  1]
# T = 10
# L = 0.705513226934028
# print(pegasos(feature_matrix, labels, T, L))

# pegasos output is ['-0.0451377', '0.0892342', '-0.0633491', '-0.0615329', '0.1293817']

# Part II


def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    labels = []
    for i in range(feature_matrix.shape[0]):
        if np.dot(theta, feature_matrix[i]) + theta_0 > 0:
            label = 1
            labels.append(label)
        else:
            label = -1
            labels.append(label)
    return np.array(labels)
    raise NotImplementedError

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    train_accuracy = 0
    val_accuracy = 0
    train_preds = []
    val_preds = []
    # Classifier is the type of classifier will be specified by user
    theta, theta_0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_preds = classify(train_feature_matrix, theta, theta_0)
    train_accuracy = accuracy(train_preds, train_labels)
    val_preds = classify(val_feature_matrix, theta, theta_0)
    val_accuracy = accuracy(val_preds, val_labels)
    return (train_accuracy, val_accuracy)
    raise NotImplementedError

def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # For this problem, loadtxt is the same as genfromtxt, both load as array
    stopwords = np.loadtxt('stopwords.txt', dtype = str)
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        # Remove the stop words from each review BEFORE creating dictionary
        # to preserve index of dictionary, *Note: word_list.remove is not correct 
        # because it only removes the 1st instance
        # __ne__ evaluates to true or false, if true, then removed from list if true
        for word in stopwords:
            word_list = list(filter((word).__ne__, word_list))
        for word in word_list:
            if word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary

# Test check: 'our', 'I', and 'their' should be removed from dictionary
# print(bag_of_words(['our I test run', 'dogy !気!2 their español@']))
# Test case 2
# texts = [
#         "He loves to walk on the beach",
#         "There is nothing better"]
# print(bag_of_words(texts))

def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                # This code gets a non-binary indicator
                feature_matrix[i, dictionary[word]] += 1
                # Original code: binary indicator [0 or 1]
                # feature_matrix[i, dictionary[word]] = 1
    return feature_matrix

#For testing
# texts = ["He loves her ",
#         "He really really loves her"]
# keys = ["he", "loves", "her", "really"]
# dictionary = {k:i for i, k in enumerate(keys)}
# print(extract_bow_feature_vectors(texts, dictionary))