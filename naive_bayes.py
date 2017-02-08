import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import TweetTokenizer


def naive_probs(ctgry_count, wrd_counts_by_ctgry):
    total = sum(ctgry_count.values())

    ctgry_probs = {}
    for ctgry in ctgry_count:
        ctgry_probs[ctgry] = ctgry_count[ctgry] / total

    wrd_probs_by_ctgry = defaultdict(lambda: defaultdict(int))
    for ctgry in ctgry_count:
        for wrd in wrd_counts_by_ctgry[ctgry]:
            wrd_probs_by_ctgry[ctgry][wrd] = \
                wrd_counts_by_ctgry[ctgry][wrd] / ctgry_count[ctgry]

    return ctgry_probs, wrd_probs_by_ctgry


def naive_classify(df, ctgry_probs, wrd_probs_by_ctgry, features):
    tknzr = TweetTokenizer()

    n = len(df.index)
    columns = ["id", "category"]
    predictions = pd.DataFrame(index=range(n), columns=columns)

    step = round(n / 100)

    for i in range(n):
        if i % step == 0:
            print("{}%".format(round(i / n * 100)))

        sentence = df['conversation'][i]
        sentence = tknzr.tokenize(sentence)
        max_prob = 0
        category = ""
        for ctgry in ctgry_probs:
            prob = ctgry_probs[ctgry]
            for w in sentence:
                if w in features:
                    prob *= wrd_probs_by_ctgry[ctgry][w]
            if prob > max_prob:
                max_prob = prob
                category = ctgry

        predictions.set_value(i, "id", df["id"][i])
        predictions.set_value(i, "category", category)
    return predictions


if __name__ == "__main__":
    data_path = "data_structures.npz"
    features_path = "features.npz"
    df_path = "test_input_clean.csv"
    save_path = df_path.replace("_clean", "_pred_naive")

    data = np.load(data_path)
    ctgry_count = data['ctgry_count'].item()
    wrd_counts_by_ctgry = data['wrd_counts_by_ctgry'].item()
    features = np.load(features_path)['keep_wrd_set'].item()

    ctgry_probs, wrd_probs_by_ctgry = \
        naive_probs(ctgry_count, wrd_counts_by_ctgry)

    df = pd.read_csv(df_path)

    predictions = naive_classify(df, ctgry_probs, wrd_probs_by_ctgry, features)
    predictions.to_csv(save_path, index=False)
