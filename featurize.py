import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer


def reduce(df, features):
    tknzr = TweetTokenizer()

    n = len(df.index)
    step = round(n / 100)

    for i in range(n):
        if i % step == 0:
            print("{}%".format(round(i / n * 100)))

        sentence = df['conversation'][i]
        sentence = tknzr.tokenize(sentence)
        sentence = [w for w in sentence if w in features]

        df.set_value(i, "conversation", " ".join(sentence))


def featurize(df, features):
    n = len(df.index)
    features = sorted(features)
    columns = ["id"] + features

    featured_df = pd.DataFrame(index=range(n), columns=columns)

    step = round(n / 100)

    for i in range(n):
        if i % step == 0:
            print("{}%".format(round(i / n * 100)))
        featured_df.set_value(i, "id", df['id'][i])
        for w in df['conversation'][i].split():
            featured_df.set_value(i, w, 1)
    return featured_df


if __name__ == "__main__":
    df_path = "train_input_clean.csv"
    features_path = "features.npz"

    df = pd.read_csv(df_path)
    features = np.load(features_path)['keep_wrd_set'].tolist()
    reduce(df, set(features))

    featured_df = featurize(df, features)
    featured_df.fillna(0, inplace=True)

    save_path = df_path.replace("_clean", "_featured")
    featured_df.to_csv(save_path, index=False)
