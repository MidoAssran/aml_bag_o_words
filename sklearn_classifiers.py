"""
SKLearn Classifiers

:description: Classifies text using sklearn classifiers
:author: Mido Assran
:date: Feb. 10, 2017
"""

import csv
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DF_TRAIN_PATH = "train_input_featured.csv"
DF_TARGET_PATH = "train_output.csv"
DF_TEST_PATH = "test_input_featured.csv"
EXPORT_PATH = "prection.csv"
EXPORT = False
DEBUG = False
NUM_DEBUG_DATA = 70000  # Number of instances to use when in DEBUG mode
# ---------------------------------------------------------------------------- #

def main():
    """ Calling code for the sklearn classifier """

    # Load training data
    data = pd.read_csv(DF_TRAIN_PATH)
    labels = pd.read_csv(DF_TARGET_PATH)
    data = data['conversation']
    labels = labels['category']

    if DEBUG:
        data = data[:NUM_DEBUG_DATA]
        labels = labels[:NUM_DEBUG_DATA]

    # Splitting into test/train sets
    X_train, X_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.001,
                                                        random_state=23)

    print('\nExtracting features from training data using sparse vectorizer')
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 max_df=0.8,
                                 stop_words='english',
                                 ngram_range=(1, 2))
    # Fit to training and testing set
    X_train = vectorizer.fit_transform(X_train, y_train)
    X_test = vectorizer.transform(X_test)

    # k = 150000
    # print('\nExtracting', k, 'best features by a chi-squared test')
    # chi_sqrd = SelectKBest(chi2, k)
    # X_train = chi_sqrd.fit_transform(X_train, y_train)
    # X_test = chi_sqrd.transform(X_test)

    # k = 100
    # print('\nExtracting', k, 'best features by a Truncated-SVD test')
    # svd = TruncatedSVD(n_components=k, random_state=42)
    # X_train = svd.fit_transform(X_train, y_train)
    # X_test = svd.transform(X_test)
    # pca = PCA(n_components=k)
    # pca.fit_transform(X_train, y_train)
    # pca.transform(X_test)

    print(X_train.shape[1])
    print("\nTraining SVM Classfier")
    # Set and fit SVM classifier
    # parameters = {'C':[1, 10, 100, 1000, 10000]}
    # svc = svm.LinearSVC()
    # clf = GridSearchCV(svc, parameters)
    # clf = svm.SVC(C=1000, gamma=0.001)
    clf = svm.LinearSVC(C=1000)
    print(clf.fit(X_train, y_train))

    # Print test score
    print(clf.score(X_train, y_train))
    print(clf.score(X_test, y_test))

    if EXPORT:
        h_predict(clf, vectorizer, None)

def h_predict(clf, vectorizer, feature_selector):
    """ Create prediction for test data """

    # Load training data
    data = pd.read_csv(DF_TEST_PATH)
    data = data['conversation']

    # Fit to training and testing set
    X_test = vectorizer.transform(data)
    if feature_selector is not None:
        feature_selector.transform(X_test)
    prediction = clf.predict(X_test)

    test = pd.DataFrame(prediction, columns=['category'])
    test.index.name = "id"
    test.to_csv(EXPORT_PATH)

if __name__ == "__main__":
    main()
