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
DS_EXPORT_PATH = "features.npz"
# ---------------------------------------------------------------------------- #

def h_chi_squared_metric(wrd_counts_by_ctgry, term, category, num_docs):
    """
    <HELPER> Compute the chi_squared_metric between 'term' and 'category'

    :ref-paper: Yang97acomparative

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type term: string
    :type category: string
    :type num_docs: int
    :rtype: float
    """
    # t is the term, c the category
    t, c = term, category
    # N is the total number of documents
    N = num_docs
    # A is the number of times t and c co-occur
    A = wrd_counts_by_ctgry[c][t]
    # B is the number of times t occurs without c
    B = sum([wrd_counts_by_ctgry[ctgry].get(t, 0)
             for ctgry in wrd_counts_by_ctgry.keys() if ctgry != c])
    # C is the number of times c occurs without t
    C = sum(wrd_counts_by_ctgry[c].values()) - A
    # D is the number of times that neither c nor t occur
    D = 0
    for ctgry in wrd_counts_by_ctgry.keys():
        if ctgry != c:
            all_wrd = sum(wrd_counts_by_ctgry[ctgry].values())
            term_wrd = wrd_counts_by_ctgry[ctgry][t]
            D += all_wrd - term_wrd
    # Return the chi_squared_metric between 'term' and 'category'
    denom = ((A + C) * (B + D) * (A + B) * (C + D))
    if denom == 0:
        return 0
    return N * (A * D - C * B) ** 2 / denom

def chi_squared_metric(wrd_counts_by_ctgry, term, num_docs, visualize=True):
    """
    Compute and visualize the chi_squared_metric between 'term' & all categories

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type term: string
    :type num_docs: int
    :rtype: void
    """
    chi_sqrd = []
    for ctgry in wrd_counts_by_ctgry.keys():
        chi_sqrd.append(
            h_chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                                 term=term, category=ctgry, num_docs=num_docs)
        )
    avg_chi_sqrd = sum(chi_sqrd) / float(len(chi_sqrd))
    if visualize:
        print("Mean Chi Squared Metric (", term, ") :", avg_chi_sqrd)
        print("Max Chi Squared Metric (", term, ") :", max(chi_sqrd))
    return avg_chi_sqrd

def doc_chi_squared_thresholding(thresh, wrd_counts_by_ctgry, num_docs):
    """
    Apply Document Frequency Thresholding to all categories

    :type thresh: int
    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type num_docs: int
    :rtype: void
    """
    i, n = 0, len(wrd_counts_by_ctgry.keys())
    chi_sqrd_by_cat = {}
    for ctgry in wrd_counts_by_ctgry.keys():
        print(round(i / n * 100), "%", end='\r')
        wrd_count = wrd_counts_by_ctgry[ctgry]
        chi_sqrd_all = [
            chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                               term=wrd,
                               num_docs=num_docs,
                               visualize=False)
            for wrd in wrd_count]
        chi_sqrd_by_cat[ctgry] = chi_sqrd_all

    for ctgry in wrd_counts_by_ctgry.keys():
        a = chi_sqrd_by_cat[ctgry]
        x = wrd_counts_by_ctgry[ctgry]
        keep_wrds = [j for (i, j) in zip(a, x) if i >= thresh]
        print(ctgry, "max:", max(a), "min:", min(a))
        wrd_counts_by_ctgry[ctgry] = Counter(
            dict([(wrd, wrd_counts_by_ctgry[ctgry][wrd]) for wrd in keep_wrds]))
    print("\n")

def visualize_frequency_metric(wrd_counts_by_ctgry, top_n=4, bottom_n=4):
    """
    Visualize the 'top_n' & 'bottom_n' most common words in each category

    :type wrd_counts_by_ctgry: dict('category': Counter('word': count))
    :type top_n: int (=DEFAULT:4)
    :type bottom_n: int (=DEFAULT:4)
    :rtype: void
    """
    print("Word Frequencies:")
    for ctgry in wrd_counts_by_ctgry.keys():
        print("\t", ctgry, ":",
              wrd_counts_by_ctgry[ctgry].most_common(top_n),
              wrd_counts_by_ctgry[ctgry].most_common()[-bottom_n:])
    print("\n")

def doc_frequency_thresholding(thresh, wrd_counts_by_ctgry):
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
    ctgry_count = data['ctgry_count'].item() # Unused for now
    num_docs = sum(ctgry_count.values())
    del ctgry_count
    # wrd_count = data['wrd_count'].item() # Unused for now
    wrd_counts_by_ctgry = data['wrd_counts_by_ctgry'].item()

    # Apply document frequency thresholding to reduce vocabulary
    thresh = 5
    doc_frequency_thresholding(thresh, wrd_counts_by_ctgry)

    # Apply document frequency thresholding to reduce vocabulary
    thresh = 40
    doc_chi_squared_thresholding(thresh, wrd_counts_by_ctgry, num_docs)


    # Visualize word frequencies in each category
    top_n, bottom_n = 4, 4
    visualize_frequency_metric(wrd_counts_by_ctgry,
                               top_n=top_n, bottom_n=bottom_n)

    # Compute the information gain

    # Wrd is an example of the term for which we want to compute chi-squared
    wrd = "people"
    # Compute the chi squared metric of each word relative to each category
    chi_squared_metric(wrd_counts_by_ctgry=wrd_counts_by_ctgry,
                       term=wrd, num_docs=num_docs)


    wrd_set = set(
        [wrd for ctgry in wrd_counts_by_ctgry.keys()
         for wrd in wrd_counts_by_ctgry[ctgry]]
    )

    # print("Keep word set:", wrd_set)

    # Export keep word set to a npz compressed file
    np.savez_compressed(DS_EXPORT_PATH,
                        keep_wrd_set=wrd_set)

if __name__ == "__main__":
    main()




