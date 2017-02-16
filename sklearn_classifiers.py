"""
SKLearn Classifiers

:description: Classifies text using sklearn classifiers
:author: Mido Assran
:date: Feb. 10, 2017
"""

import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from collections import Counter
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DF_TRAIN_PATH = "train_input_clean.csv"
DF_TARGET_PATH = "train_output.csv"
DF_TEST_PATH = "test_input_clean.csv"
EXPORT_PATH = "prection.csv"
EXPORT = False
DEBUG = False
VISUALIZE = True
NUM_DEBUG_DATA = 10000  # Number of instances to use when in DEBUG mode
# ---------------------------------------------------------------------------- #

def main():
    """ Calling code for the sklearn classifier """

    print(__doc__)


    # Load training data
    dataone = pd.read_csv(DF_TRAIN_PATH)
    labelsone = pd.read_csv(DF_TARGET_PATH)
    data = dataone['conversation']
    labels = labelsone['category']

    if DEBUG:
        data = data[:NUM_DEBUG_DATA]
        labels = labels[:NUM_DEBUG_DATA]

    # Splitting into test/train sets
    X_train, X_test_ren, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=23)


    print('\nExtracting features from training data using sparse vectorizer')
    vectorizer = TfidfVectorizer(sublinear_tf=True,
                                 max_df=0.8,
                                 stop_words='english',
                                 ngram_range=(1, 4))
    # Fit to training and testing set
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test_ren)

    print("Num. Features:", X_train.shape[1])

    print("\nTraining SVM Classfier")
    clf = svm.LinearSVC(C=240)
    clf.fit(X_train, y_train)

    # Print accuracy
    print("Train:", clf.score(X_train, y_train))
    print("Test:", clf.score(X_test, y_test))

    # --------------- Flow 2 -----------
    data2 = dataone[(labelsone['category'] == 'worldnews')
                    | (labelsone['category'] == 'politics')
                    | (labelsone['category'] == 'news')]['conversation']
    labels2 = labelsone[(labelsone['category'] == 'worldnews')
                    | (labelsone['category'] == 'politics')
                    | (labelsone['category'] == 'news')]['category']

    if DEBUG:
        data2 = data2[:NUM_DEBUG_DATA]
        labels2 = labels2[:NUM_DEBUG_DATA]

    print('\nExtracting features from training data using sparse vectorizer')
    # Fit to training and testing set
    X_train2 = vectorizer.transform(data2)
    y_train2 = labels2


    print("Num. Features:", X_train2.shape[1])
    clf2 = MultinomialNB(alpha=0.0001)
    clf2.fit(X_train2, y_train2)
    print(clf2.score(X_train2, y_train2))

    # Replace the predictions in SVM by naive
    p1 = clf.predict(X_test)
    df1 = pd.DataFrame(p1, columns=['category'])
    df1['conversation'] = X_test_ren.values

    data3 = df1[(df1['category'] == 'worldnews')
                    | (df1['category'] == 'politics')
                    | (df1['category'] == 'news')]['conversation']
    labels3 = df1[(df1['category'] == 'worldnews')
                    | (df1['category'] == 'politics')
                    | (df1['category'] == 'news')]['category']


    print('\nExtracting features from training data using sparse vectorizer')
    # Fit to training and testing set
    X_test3 = vectorizer.transform(data3)
    # Naive predictions on the hockey, worldnes predictions of svm
    p2 = clf2.predict(X_test3)


    print("TOTAL old score:", 
            metrics.accuracy_score(y_test, df1['category'].tolist()))

    if VISUALIZE:
        y_pred = df1['category'].tolist()
        # class_names = Counter(y_pred).keys()
        class_names = ['nfl', 'worldnews', 'hockey', 'news', 'politics', 'soccer', 'movies', 'nba']
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred, class_names)
        h_visualize(cnf_matrix, classes=class_names,
                    title='Confusion matrix, without normalization')

    df1.loc[(df1['category'] == 'worldnews')
            | (df1['category'] == 'politics')
            | (df1['category'] == 'news'), 'category'] = p2


    print("TOTAL new score:", 
            metrics.accuracy_score(y_test, df1['category'].tolist()))

    if VISUALIZE:
        y_pred = df1['category'].tolist()
        # class_names = Counter(y_pred).keys()
        class_names = ['nfl', 'worldnews', 'hockey', 'news', 'politics', 'soccer', 'movies', 'nba']
        cnf_matrix = metrics.confusion_matrix(y_test, y_pred, class_names)
        h_visualize(cnf_matrix, classes=class_names,
                    title='Confusion matrix, without normalization')

    if EXPORT:
        h_predict(clf, vectorizer, None)

def h_visualize(cm, classes,
                normalize=False,
                title='Confusion matrix',
                cmap=plt.cm.Blues):
    cm = cm
    classes = classes
    title = title
    cmap = cmap

    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cm, classes=classes,
                          title='Confusion matrix, without normalization')

    plt.show()


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
