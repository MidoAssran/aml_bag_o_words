import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pandas as pd
from math import sqrt
import warnings
from collections import Counter
import random
import time
import copy
import heapq
import matplotlib.pyplot as plt
import itertools
start_time = time.time()

"""
Author: Ziad Saliba 
K Nearest Neighbors local implementation as well as sk-learn implementation
contact: ziad.saliba@mail.mcgill.ca
reference: Sentdex youtube chanel https://www.youtube.com/playlist?list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v videos #13-19, used for base idea of this implentation
"""

##----------File import paths------------##
IMRORT_PATH_X = 'train_input_featured_005_30k.csv'
IMPORT_PATH_Y = 'train_output_30k.csv'
##---------------------------------------##

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    ref: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],2)
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

##----------------kNN Classifier Function-----------##
#@param - data: training set
#@param - predict: test set
#@param - k: number of neighbors
def k_NN(data, predict, k):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups!')
    dist = []
    for group in data:
        for feautures in data[group]:
            neg_euclid_dist = -np.linalg.norm(np.array(feautures) - np.array(predict))
            if len(dist) < k:
                heapq.heappush(dist, (neg_euclid_dist, group))
            elif neg_euclid_dist > dist[0][0]:
                heapq.heapreplace(dist, (neg_euclid_dist, group))

    votes =[votes[1] for votes in dist]
    vote_result = Counter(votes).most_common(1)[0][0]
    #confidence = Counter(votes).most_common(1)[0][1] / k    ##confidence provides the majority fraction of votes

    return vote_result #,confidence

##----------------Sklearn kNN Function -----------##
#@param - X_train: training features
#@param - y_train: training lables
#@param - X_test: validation features
#@param - y_test: validation labels
def sk_k_NN(X_train, y_train, X_test, y_test, class_names):
    #k_values = {'n_neighbors':[5,10,15,20]}
    #clf = GridSearchCV(neighbors.KNeighborsClassifier(), k_values, n_jobs = 3)
    clf = neighbors.KNeighborsClassifier(5, n_jobs = 3)
    print("Fitting Sklearn classifier...")
    clf.fit(X_train, y_train)
    print("\nTesting Sklearn classifier...")
    preds = np.array(clf.predict(X_test))
    y_test = np.array(y_test)
    correct = np.where(preds==y_test,1,0)
    accuracy = np.sum(correct)/len(y_test)
    print(classification_report(y_test, preds, target_names = class_names))

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, preds)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()

    return accuracy     

##----------------Function to fill training & testing dictionaries -----------##
#@param - train: training dataframe
#@param - test: testing dataframe
#@param - train_set: training dictionary
#@param - test_test: validation dictionary
def make_dicts(train, test, train_set, test_set):

    train_data = train.values.tolist()
    test_data = test.values.tolist()
    
    for i in train_data:
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        test_set[i[-1]].append(i[:-1])
        
def main():

    #read input file
    df = pd.read_csv(IMRORT_PATH_X)
    #fill nan values with 0
    df.fillna(0, inplace = True)
    #read output values
    labels = pd.read_csv(IMPORT_PATH_Y)
    #drop id column from df
    df.drop(['id'],1,inplace = True)
    #add labels to main dataframe
    df['category'] = labels['category']
    
    train, test = cross_validation.train_test_split(df, test_size = 0.2, random_state = 34)

    #training and testing dictionaries hashed by category key
    train_set = {'news':[], 'nfl':[], 'nba':[], 'hockey':[], 'soccer':[], 'worldnews':[], 'politics':[], 'movies':[]}
    test_set = {'news':[], 'nfl':[], 'nba':[], 'hockey':[], 'soccer':[], 'worldnews':[], 'politics':[], 'movies':[]}

    make_dicts(train, test, train_set, test_set)
        
    correct = 0
    total = 0
    count = 1
    y_pred = []
    class_names = ['news','nfl','hockey','nba','soccer','worldnews','politics','movies']

    ##iterates over test set and calls local kNN implementation function to make predictions
    for group in test_set:
        for data in test_set[group]:
            vote = k_NN(train_set, data, 5)
            y_pred.append(vote)
            print("Predicted: ", count, " out of ", len(test), "\t", round((count/(len(test)))*100,2),"% done.")
            count +=1
            if group == vote:
                correct += 1
            total += 1
        
    print('\nLocal kNN Accuracy: ', correct/total)
    print('\nSklearn kNN Accuracy: ', sk_k_NN(train.ix[:, train.columns != 'category'], train['category'], test.ix[:, test.columns != 'category'], test['category'], class_names))
    print("\n--- %s seconds ---" % round((time.time() - start_time)))

if __name__ == "__main__":
    main()
