import pandas as pd
import numpy as np
from nltk.tokenize import TweetTokenizer


def reduce(df, features):
    tknzr = TweetTokenizer()

    n = len(df.index)
    step = round(n / 100)

    for i in range(n):
        if i % step == 0:
            print("{}%".format(round(i / n * 100)), end='\r')

        sentence = df['conversation'][i]
        sentence = tknzr.tokenize(sentence)
        sentence = [w for w in sentence if w in features]

        df.set_value(i, "conversation", " ".join(sentence))


if __name__ == "__main__":
    df_path = "test_input_clean.csv"
    features_path = "features.npz"

    df = pd.read_csv(df_path)
    features = np.load(features_path)['keep_wrd_set'].tolist()
    reduce(df, set(features))
    print("\nNum features:", len(features))
    save_path = df_path.replace("_clean", "_featured")
    df.to_csv(save_path, index=False)
