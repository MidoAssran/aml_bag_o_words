"""
Data Structure Creation Script

:description: Converts redit conversation data into fast-access data structures
:author: Mido Assran
:date: Feb. 4, 2017
:note: Function names prepended with 'h_' are helper functions
"""

from collections import Counter
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DF_TRAIN_PATH = "train_input_clean.csv"
DF_TARGET_PATH = "train_output.csv"
DS_EXPORT_PATH = "data_structures.npz"
# ---------------------------------------------------------------------------- #

def main():
    """ Calling code for the data structure creator """


    # Create a single unified data frame with targets
    dframe = pd.read_csv(DF_TRAIN_PATH)
    dframe['category'] = pd.read_csv(DF_TARGET_PATH)['category']

    # Tokenize all the conversations
    dframe['tknzd_conversation'] = dframe['conversation'].apply(word_tokenize)

    # Set of all categories in dframe
    ctgry_count = Counter(list(dframe['category']))

    # Set of the words in the dframe organized by category
    wrd_counts_by_ctgry = {}
    for ctgry in ctgry_count:
        ctgry_dframe = dframe[dframe['category'] == ctgry]
        cnvs = list(ctgry_dframe['tknzd_conversation'])
        wrd_lst = [item for sublist in cnvs for item in sublist]
        wrd_counts_by_ctgry[ctgry] = Counter(wrd_lst)

    # Set of all words in dframe
    wrd_count = sum(wrd_counts_by_ctgry.values(), Counter())

    # Export results to a npz compressed file
    np.savez_compressed(DS_EXPORT_PATH,
                        wrd_count=wrd_count,
                        ctgry_count=ctgry_count,
                        wrd_counts_by_ctgry=wrd_counts_by_ctgry)


if __name__ == "__main__":
    main()




