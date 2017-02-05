"""
Word Redux

:description: Reduces vocabulary based on data metrics
:author: Mido Assran
:date: Feb. 4, 2017
:note: Function names prepended with 'h_' are helper functions
"""

from collections import Counter
from itertools import dropwhile
import numpy as np

# ---------------------------------------------------------------------------- #
# Script Parameters
# ---------------------------------------------------------------------------- #
DS_LOAD_PATH = "data_structures.npz"
# ---------------------------------------------------------------------------- #

def h_chi_squared_metric(wrd_counts_by_ctgry, ctgry_count, term, category):
    """
    <HELPER> Compute the chi_squared_metric between 'term' and 'category'

    :ref-paper: Yang97acomparative

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
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
    B = sum([wrd_counts_by_ctgry[ctgry].get(t, 0)
             for ctgry in ctgry_count if ctgry != c])
    # C is the number of times c occurs without t
    C = sum(wrd_counts_by_ctgry[c].values()) - A
    # D is the number of times that neither c nor t occur
    D = 0
    for ctgry in ctgry_count:
        if ctgry != c:
            all_wrd = sum(wrd_counts_by_ctgry[ctgry].values())
            term_wrd = wrd_counts_by_ctgry[ctgry][t]
            D += all_wrd - term_wrd
    # Return the chi_squared_metric between 'term' and 'category'
    denom = ((A + C) * (B + D) * (A + B) * (C + D))
    if denom == 0:
        return 0
    return N * (A * D - C * B) ** 2 / denom

def visualize_chi_squared_metric(wrd_counts_by_ctgry, ctgry_count, term):
    """
    Compute and visualize the chi_squared_metric between 'term' & all categories

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
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
                                 term=term, category=ctgry)
        )
    avg_chi_sqrd = sum(chi_sqrd) / float(len(chi_sqrd))
    print("Mean Chi Squared Metric (", term, ") :", avg_chi_sqrd)
    print("Max Chi Squared Metric (", term, ") :", max(chi_sqrd))


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
    print("Category frequencies:\n\t", ctgry_counts.most_common(10), end="\n\n")
    print("Word Frequencies:")
    for ctgry in ctgry_counts:
        print("\t", ctgry, ":",
              wrd_counts_by_ctgry[ctgry].most_common(top_n),
              wrd_counts_by_ctgry[ctgry].most_common()[-bottom_n:])
    print("\n")

def document_frequency_thresholding(thresh, wrd_counts_by_ctgry):
    """
    Apply Document Frequency Thresholding to all categories

    :type thresh: int
    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :rtype: void
    """
    for wrd_count in wrd_counts_by_ctgry.values():
        for wrd, _ in dropwhile(lambda key_count: key_count[1] >= thresh, wrd_count.most_common()):
            del wrd_count[wrd]

def main():
    """ Calling code for the word redux """

    data = np.load(DS_LOAD_PATH)
    ctgry_count = data['ctgry_count'].item()
    # wrd_count = data['wrd_count'].item() # Unused for now
    wrd_counts_by_ctgry = data['wrd_counts_by_ctgry'].item()

    # Apply document frequency thresholding to reduce vocabulary
    thresh = 15
    document_frequency_thresholding(thresh, wrd_counts_by_ctgry)

    # Visualize word frequencies in each category
    top_n, bottom_n = 2, 2
    visualize_frequency_metric(ctgry_count, wrd_counts_by_ctgry,
                               top_n=top_n, bottom_n=bottom_n)

    # Compute the information gain

    # Wrd is an example of the term for which we want to compute chi-squared
    wrd = "gulfnews"
    # Compute the chi squared metric of each word relative to each category
    visualize_chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                                 ctgry_count=ctgry_count,
                                 term=wrd)



if __name__ == "__main__":
    main()




