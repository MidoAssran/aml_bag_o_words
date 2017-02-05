"""
Visualizer.py

:description: Evaluates lexical metrics on sanitized redit conversation data
:author: Mido Assran
:date: Feb. 4, 2017
:note: Function names prepended with 'h_' are helper functions
"""

import numpy as np
from collections import Counter

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DS_LOAD_PATH = "data_structures.npz"
# ---------------------------------------------------------------------------- #

def h_chi_squared_metric(wrd_counts_by_ctgry, ctgry_counts_by_wrd, ctgry_count,
                         term, category):
    """
    <HELPER> Compute the chi_squared_metric between 'term' and 'category'

    :ref-paper: Yang97acomparative

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type ctgry_counts_by_wrd: dict('word': Counter('category': count))
    :type ctgry_count: Counter('category': count)
    :type top_n: int (=DEFAULT:4)
    :type bottom_n: int (=DEFAULT:4)
    :rtype: float
    """
    # t is the term, c the category
    t, c = term, category
    # N is the total number of documents
    N = sum(ctgry_count.values())
    # A is the number of times t and c co-occur
    A = wrd_counts_by_ctgry[c][t]
    # B is the number of times t occurs without c
    B = sum(ctgry_counts_by_wrd[t].values()) - A
    # C is the number of times c occurs without t
    C = sum(wrd_counts_by_ctgry[c].values()) - A
    # D is the number of times that neither c nor t occur
    D = N - (A + B + C)
    # Return the chi_squared_metric between 'term' and 'category'
    return N * (A * D - C * B) ** 2 / ((A + C) * (B + D) * (A + B) * (C + D))

def visualize_chi_squared_metric(wrd_counts_by_ctgry, ctgry_counts_by_wrd,
                                 ctgry_count, term):
    """
    Compute and visualize the chi_squared_metric between 'term' & all categories

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type ctgry_counts_by_wrd: dict('word': Counter('category': count))
    :type ctgry_count: Counter('category': count)
    :type top_n: int (=DEFAULT:4)
    :type bottom_n: int (=DEFAULT:4)
    :rtype: void
    """
    chi_sqrd = []
    for ctgry in ctgry_count:
        chi_sqrd.append(
            h_chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                                 ctgry_count=ctgry_count,
                                 ctgry_counts_by_wrd=ctgry_counts_by_wrd,
                                 term=term, category=ctgry)
        )
    avg_chi_sqrd = sum(chi_sqrd) / float(len(chi_sqrd))
    print("Chi Squared Metric (", term, ") :", avg_chi_sqrd)

def visualize_frequency_metric(ctgry_counts, wrd_counts_by_ctgry,
                               top_n=4, bottom_n=4):
    """
    Visualize the 'top_n' & 'bottom_n' most common words in each category

    :type ctgry_count: Counter('category': int)
    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type top_n: int (=DEFAULT:4)
    :type bottom_n: int (=DEFAULT:4)
    :rtype: void
    """
    print("Category frequencies:", ctgry_counts, end="\n\n")
    print("Word Frequencies:")
    for ctgry in ctgry_counts:
        print("\t", ctgry, ":",
              wrd_counts_by_ctgry[ctgry].most_common(top_n),
              wrd_counts_by_ctgry[ctgry].most_common()[-bottom_n:])

def main():
    """ Calling code for the visualizer """

    data = np.load(DS_LOAD_PATH)
    ctgry_count = data['ctgry_count'].item()
    # wrd_count = data['wrd_count'].item() # Unused for now
    wrd_counts_by_ctgry = data['wrd_counts_by_ctgry'].item()
    ctgry_counts_by_wrd = data['ctgry_counts_by_wrd'].item()


    # Visualize word frequencies in each category
    visualize_frequency_metric(ctgry_count, wrd_counts_by_ctgry,
                               top_n=2, bottom_n=2)

    # Compute the information gain

    # Wrd is an example of the term for which we want to compute chi-squared
    wrd = "goal"
    # Compute the chi squared metric of each word relative to each category
    visualize_chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                                 ctgry_count=ctgry_count,
                                 ctgry_counts_by_wrd=ctgry_counts_by_wrd,
                                 term=wrd)


if __name__ == "__main__":
    main()




