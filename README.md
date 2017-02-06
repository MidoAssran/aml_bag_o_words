# Reddit Conversation Classifier

Several steps must be taken to prepare the data for training. The following assumes that the files 'train_input.csv' and 'train_output.csv' are present in the local directory:

1. Run **sanitizer.py**. This removes html tags, stop words and punctuation from the original dataset. It saves the results as create 'train_input_clean.csv'.
```
python sanitizer.py
```
2. Run **data_structure_creator.py**. This will create a set of dictionaries analyzing the distribution of words in the dataset, and save the results as 'data_structures.npz'.
```
python data_structure_creator.py
```
3. Run **word_redux.py**. This will extract the words to be used as features from the distribution file. It saves the results as 'features.npz'
```
python word_redux.py
```
4. Finally, run **featurize.py**. This will create a new csv where each column represents a feature.
```
python featurize.py
```
