import pandas as pd 
from sklearn import preprocessing, cross_validation, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
from sklearn.model_selection import train_test_split

##----------File import paths------------##
IMRORT_PATH_X = 'train_input.csv'
IMPORT_PATH_Y = 'train_output.csv'
##---------------------------------------##

###ref: http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html

#reading data, this is raw input, not sanitized
data = pd.read_csv(IMRORT_PATH_X)
labels = pd.read_csv(IMPORT_PATH_Y)

data = data['conversation']
labels = labels['category']

#splitting into test/train sets
X_train, X_test, y_train, y_test, = train_test_split(data, labels, test_size = 0.2, random_state = 23)

 
print('\nExtracting features from training data using a sparse vectorizer')

#get rid of english stopwords and words that appear in more than 80% of the conversation 
vectorizer = TfidfVectorizer(sublinear_tf = True, stop_words = 'english', max_df = 0.8)

#fit to training, then apply on testing
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

##CHOOSE NUMBER OF FEATURES##
k = 3750
##-------------------------##

print('\nExtracting', k, 'best features by a chi-squared test')

#select k-best features based of chi2 testing
chi2 = SelectKBest(chi2, k)

#fit to training, apply on testing
X_train = chi2.fit_transform(X_train, y_train)
X_test = chi2.transform(X_test)

#set and fit svm classifier, gamma and C values taken from a colleague...
clf = svm.SVC(gamma = 0.001, C = 140)
clf.fit(X_train, y_train)

#predict on training and testing
train_pred = clf.predict(X_train)
test_pred = clf.predict(X_test)

#check training and testing accuracy
test_score = metrics.accuracy_score(y_test, test_pred)
train_score = metrics.accuracy_score(y_train, train_pred)

#print training and testing accuracy
print('\nTraining accuracy: ', train_score) 
print('\nTesting accuracy: ', test_score)

###---the settings in this file achieved a train score of ~0.89 and a test score of ~0.88---###