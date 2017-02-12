# Reddit Conversation Classifier

## Requirements

The following python packages are required:
* numpy
* pandas
* scikit-learn
* nltk

In addition, once nltk is installed you must run the following commands (in a python session) to install the required corpora and models: `nltk.download("stopwords")`, `nltk.download("punkt")`, and `nltk.download("wordnet")`.


## Data Preparation

Several steps must be taken to prepare the data for training. The following assumes that the files 'train_input.csv' and 'train_output.csv' are present in the local directory:

1. Run `python sanitizer.py`. This removes html tags, stop words and punctuation from the original dataset. It saves the results as create 'train_input_clean.csv'.
2. Run `python data_structure_creator.py`. This will create a set of dictionaries analyzing the distribution of words in the dataset, and save the results as 'data_structures.npz'.
3. Run `python word_redux.py`. This will extract the words to be used as features from the distribution file. It saves the results as 'features.npz'.
4. Finally,
    a) for local ML implementations, run `python featurize.py`. This will create a new dataframe where each column represents a feature. It saves the results as 'train_input_featured.csv'.
    b) Otherwise for sklean_classifier implementations, run 'python data_frame_reduce.py'. This will simply filter the training csv data to keep only the words specified by 'word_redux.py'. The results are saved as 'train_input_featured.csv'


