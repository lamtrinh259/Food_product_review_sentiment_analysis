# Amazon_food_product_review_sentiment_analysis
Classify Amazon's food product review to see whether the review was positive (1 value) or negative (-1 value) using linear classifiers built from scratch. The algorithms used are all binary classifiers.

[The live version of this app is here, you can give it a try by inputting your own review.](https://share.streamlit.io/lamtrinh259/food_product_review_sentiment_analysis/main/review_sentiment_app.py)

These are the steps that I took to deliver this project: 
1. Importing necessary libraries and build some supporting functions: hinge loss to calculate the loss value of each point. 
2. I built the single-step version of each of these algorithms to solve for Support Vector Machine for one feature and one label: Perceptron, Average Perceptron, and Pegasos. 
3. Then I built a loop to apply the single-step version T * n times, where T is inputted by user and n is the length of the dataset (equivalent to the number of product reviews). 
4. Then I built a classify function to classify whether the data point (review) was negative or positive. 
5. A classifier accuracy function was then built to measure the accuracy of the algorithm. 
6. I train the 3 models on a training datasets of 4,000 food product reviews. These trained models were validated on 500 product reviews (validation dataset) in order to select the model with the best accuracy (in this case it was Pegasos). 
7. Afterwards, the best model (Pegasos) was then used to classify 500 unseen food product (test dataset). 
8. Overall, the best algorithm was **Pegasos** with **80.6% accuracy** with tuned parameters. 
9. In the end, I experimented with some feature engineering such as including in stop words (words that should be removed from the "dictionary") and changing binary features to counts features. 

*Important note: the review texts used were converted into feature vectors by using a **bag of word** approach. All the words that appear in a training set of reviews were converted into a dictionary, thus producing a list of d unique words. Each of the reviews is then transformed into a feature vector of length i by setting the i_th coordinate of the feature vector to 1 if the i_th word in the dictionary appears in the review, or 0 otherwise. For instance, consider two simple reviews “I love this product!" and “The taste is quite bland". In this case, the dictionary is the set {I; love; this; product; !; the; taste; is; quite; bland}, and the reviews are represented as {1, 1, 1, 1, 1, 0, 0, 0, 0, 0} and {0, 0, 0, 0, 0, 1, 1, 1, 1, 1}. Keep in mind that numbers (0~9) and punctuation (!, ?, etc.) are treated as their own words in the dictionary. Furtheremore, a _unigram model_ was used where each word is a distinct key in the dictionary. 

Files: 
- project1.py is where I implemented the learning algorithms.
- main.py is a script skeleton where the functions in project1 are called on and I run this to see the results and experiment. 
- utils.py contains utility functions to support throughout. 
- test.py is a script which runs tests on a few of the methods that were implemented. 
- reviews_train.tsv : training dataset
- reviews_val.tsv: validation dataset
- reviews_test.tsv: test dataset
- stopwords.txt: the list of stop words that I used

For further development or experimentation, one can: 
- Use a bigram model where a pair of words is a distinct key in the dictionary. 
- How to treat occurrence of all-cap words (e.g., 'THIS IS AMAZING!')
- What to do with length of the text. E.g. if a text is long, is that generally a good review, or bad review? 
- Incorporate the number of helpful upvotes or downvotes into the weight vectors if they're available. 
