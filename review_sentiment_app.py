# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 10:30:47 2021

@author: Lam
"""
import csv
from string import punctuation, digits
import random
import numpy as np
# import pandas as pd
import streamlit as st
import sys

# Note: I only need to cache any functions for the training part, for the testing
# part (user input), it will run again from beginning
# All the necessary functions for this to run on its own
# This is to determine Python version
if sys.version_info[0] < 3:
    PYTHON3 = False
else:
    PYTHON3 = True

# Load data function, which relies on csv and sys
@st.cache(persist = True) # Memorize the result in cache so that re-run won't take much time
def load_data(path_data, extras=False):
    """
    Returns a list of dict with keys:
    * sentiment: +1 or -1 if the review was positive or negative, respectively
    * text: the text of the review

    Additionally, if the `extras` argument is True, each dict will also include the
    following information:
    * productId: a string that uniquely identifies each product
    * userId: a string that uniquely identifies each user
    * summary: the title of the review
    * helpfulY: the number of users who thought this review was helpful
    * helpfulN: the number of users who thought this review was NOT helpful
    """

    global PYTHON3

    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    if PYTHON3:
        f_data = open(path_data, encoding="latin1")
    else:
        f_data = open(path_data)

    for datum in csv.DictReader(f_data, delimiter='\t'):
        for field in list(datum.keys()):
            if not extras and field not in basic_fields:
                del datum[field]
            elif field in numeric_fields and datum[field]:
                datum[field] = int(datum[field])

        data.append(datum)

    f_data.close()

    return data

# Extract words function, which relies on string.punctuation + digits
@st.cache
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

# Bag-of-words function, which uses extract-word function above
@st.cache
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # For this problem, loadtxt is the same as genfromtxt, both load as array
    stopwords = np.loadtxt('stopwords.txt', dtype = str)
    # Trying out bigram feature, mapping 2 consecutive words as key in dict
    dictionary = {} # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        # Switch position of filter before creating word_list_pairs
        for word in stopwords:
            word_list = list(filter((word).__ne__, word_list))
        # Note that word_list_pairs is a generator
        word_list_pairs = [(word1, word2) for (word1, word2) in zip(word_list[:-1], word_list[1:])]
        # __ne__ evaluates to true or false, if true, then removed from list if true
        for (word1, word2) in word_list_pairs:
            if (word1, word2) not in dictionary:
                dictionary[(word1, word2)] = len(dictionary)
    return dictionary

@st.cache
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
    stopwords = np.loadtxt('stopwords.txt', dtype = str)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in stopwords:
            word_list = list(filter((word).__ne__, word_list))
        word_list_pairs = [(word1, word2) for (word1, word2) in zip(word_list[:-1], word_list[1:])]
        for (word1, word2) in word_list_pairs:
            if (word1, word2) in dictionary:
                # This code gets a non-binary indicator
                feature_matrix[i, dictionary[(word1, word2)]] += 1
    return feature_matrix

# This function depends on random library
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

# Pegasos algorithm, which depends on get_order function above
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

    if label*(np.dot(feature_vector, current_theta) + current_theta_0) <= 1:
        theta = (1-eta*L)*current_theta + eta*label*feature_vector
        theta_0 = current_theta_0 + eta*label 
    else: 
        theta = (1-eta*L)*current_theta
        theta_0 = current_theta_0
    return (theta, theta_0)

# This depends on the single-step Pegasos algorithm above
@st.cache
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

# Classification function
@st.cache
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


st.write("""
# Simple Amazon food product review Sentiment app

Please enter a review of any product (Amazon food product or not) in the text
box below. Shortly afterwards, the machine learning algorithm (Pegasos solver
algorithm for Support Vector Machine) will evaluate whether the review 
sentiment is positive or negative.

*Note: the 1st time the app is run, it may take a few minutes to run because
it needs to train itself on the existing data. From the 2nd time onwards, it will
run much faster!* 
""")

st.markdown("For example, this review below would be categorized as *'positive'*:")
st.markdown("The food is delicious!")
st.markdown("In contrast, this review below would be categorized as *'negative'*:")
st.markdown("The drink tastes horrible, I wasted my money.")

# Create a text input box where user can type in the review
review_input = st.text_input('Product review', 'This product is great!')

# Reprint the review that user put in so they can see
# st.write("The current product review is '" + review_input +"'")

# Data loading
train_data = load_data('reviews_train.tsv')
val_data = load_data('reviews_val.tsv')
# Test data is what the user input in
test_data = review_input

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
# Adding the review(s) to list since the extract bow function only works with list
test_texts = []
test_texts.append(test_data)

dictionary = bag_of_words(train_texts)

train_bow_features = extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = extract_bow_feature_vectors(test_texts, dictionary)

# Get the theta for the algorithm with default parameters
theta, theta_0 = pegasos(train_bow_features, train_labels, T = 25, L = 0.01)

# Predict whether the review is positive or negative
review_pred = classify(test_bow_features, theta, theta_0)

# Print out the review sentiment
if review_pred > 0:
    st.write("The sentiment of your product review is **positive**")
else:
    st.write("The sentiment of your product review is **negative**")