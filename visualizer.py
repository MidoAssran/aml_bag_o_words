"""
Visualizer.py

:description: Evaluates lexical metrics on sanitized redit conversation data
:author: Mido Assran
:date: Feb. 4, 2017
:note: Function names prepended with 'h_' are helper functions
"""

from collections import Counter
import pandas as pd
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DF_TRAIN_PATH = "train_input_clean.csv"
DF_TARGET_PATH = "train_output.csv"
# ---------------------------------------------------------------------------- #

def h_category_counts(dframe, ctgry_set):
    """
    <Helper> Determine the number of instances of each category in the dframe

    :type dframe: pandas.dframe
    :type ctgry_set: set(str)
    :rtype: dict("category": count)
    """
    ctgry_counts = {}
    for ctgry in ctgry_set:
        ctgry_counts[ctgry] = dframe[dframe['category'] == ctgry].count()[0]
    return ctgry_counts

def h_word_counts_by_category(ctgry_set, wrd_lst_by_ctgry):
    """
    <Helper> Determine the number of words in each category of the dframe

    :type ctgry_set: set(str)
    :type wrd_lst_by_ctgry: dict("category": list(str))
    :rtype: dict("category": dict("word": count))
    """
    counts = {}
    for ctgry in ctgry_set:
        wrd_lst = wrd_lst_by_ctgry[ctgry]
        counts[ctgry] = Counter(wrd_lst)
    return counts

def frequency_metric(ctgry_counts, wrd_counts, top_n=4):
    """
    Visualize the frequency of the 'top_n' most common words in each category

    :type top_n: int (=DEFAULT:4)
    :rtype: void
    """
    print("Category frequencies:", ctgry_counts)
    print("Word Frequencies:")
    for ctgry in ctgry_counts:
        print("\t", ctgry, ":", wrd_counts[ctgry].most_common(top_n))

def main():
    """ Calling code for the visualizer """

    # ------------------------------------------ #
    # Pre-calculations
    # ------------------------------------------ #
    # Create a single unified data frame with targets
    dframe = pd.read_csv(DF_TRAIN_PATH)
    dframe['category'] = pd.read_csv(DF_TARGET_PATH)['category']

    # Data structure of the categories in the dframe
    ctgry_set = set(list(dframe['category']))

    # Data structure of the words in each category in the dframe
    wrd_lst_by_ctgry = {}
    for ctgry in ctgry_set:
        ctgry_dframe = dframe[dframe['category'] == ctgry]
        cnvs = list(ctgry_dframe['conversation'])
        wrd_lst = []
        for cnv in cnvs:
            wrd_lst += word_tokenize(cnv)
        wrd_lst_by_ctgry[ctgry] = wrd_lst

    # Distribution of categories
    ctgry_counts = h_category_counts(dframe, ctgry_set)
    # Distribution of words by category
    wrd_counts = h_word_counts_by_category(ctgry_set, wrd_lst_by_ctgry)
    # ------------------------------------------ #

    # Visualize word frequencies in each category
    frequency_metric(ctgry_counts, wrd_counts, top_n=6)

    # Compute the information gain


    # Compute the chi squared metric of each word relative to each category



if __name__ == "__main__":
    main()




