"""
Visualizer.py

:description: Evaluates lexical metrics on sanitized redit conversation data
:author: Mido Assran
:date: Feb. 4, 2017
"""

import pandas as pd
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DF_TRAIN_PATH = "train_input_clean.csv"
DF_TARGET_PATH = "train_output.csv"
# ---------------------------------------------------------------------------- #

def category_frequencies(dframe, ctgry_set):
    """
    Determine the number of instances of each category in the dframe

    :type dframe: pandas.dframe
    :type ctgry_set: set(str)
    :rtype: dict("category": count)
    """
    ctgry_counts = {}
    for cat in ctgry_set:
        ctgry_counts[cat] = dframe[dframe['category'] == cat].count()[0]
    return ctgry_counts

def word_frequencies_by_category(ctgry_set, wrd_lst_by_ctgry):
    """
    Determine the frequency of words in each category of the dframe

    :type ctgry_set: set(str)
    :type wrd_lst_by_ctgry: dict("category": list(str))
    :rtype: dict("category": dict("word": count))
    """
    counts = {}
    for cat in ctgry_set:
        wrd_lst = wrd_lst_by_ctgry[cat]
        counts[cat] = {}
        unique_words = set(wrd_lst)
        for unq_wrd in unique_words:
            counts[cat][unq_wrd] = wrd_lst.count(unq_wrd)
    return counts

def main():
    """ Calling code for the visualizer """

    # Create a single unified data frame with targets
    dframe = pd.read_csv(DF_TRAIN_PATH)
    dframe['category'] = pd.read_csv(DF_TARGET_PATH)['category']

    # Set of categories in dframe
    ctgry_set = set(list(dframe['category']))

    # List of words in each category
    wrd_lst_by_ctgry = {}

    id_list = list(dframe['id'])
    n_samples = len(id_list)
    for uid in id_list:

        print("{}%".format(round(uid / n_samples * 100)), end='\r')

        ctgry = str(dframe[dframe['id'] == uid]['category'])
        if ctgry not in wrd_lst_by_ctgry:
            wrd_lst_by_ctgry[ctgry] = []

        conv = str(dframe[dframe['id'] == uid]['conversation'])
        wrd_lst_by_ctgry[ctgry] += word_tokenize(conv)


    # Distribution of data samples by category
    ctgry_counts = category_frequencies(dframe, ctgry_set)
    # Distribution of words by category
    wrd_counts = word_frequencies_by_category(ctgry_set=ctgry_set,
                                              wrd_lst_by_ctgry=wrd_lst_by_ctgry)
    print(ctgry_counts)
    print(wrd_counts['hockey'])

if __name__ == "__main__":
    main()
