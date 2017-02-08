import re
import string
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer


def sanitize(df, stem=False):
    tknzr = TweetTokenizer()
    st = LancasterStemmer()
    stops = set(stopwords.words('english'))
    letters = set(string.ascii_lowercase)
    others = {"com", "www", "imgur", "org", "net"}
    unwanted = stops | letters | others

    n = len(df.index)
    step = round(n / 100)

    for i in range(n):
        if i % step == 0:
            print("{}%".format(round(i / n * 100)))

        sentence = df['conversation'][i]
        sentence = re.sub(r"<.*?>", "", sentence)  # remove html tags
        sentence = re.sub(r"@.*?\s", "", sentence)  # remove @ mentions
        sentence = re.sub(r"[0-9]", "", sentence)  # remove ALL numbers
        sentence = re.sub(r"[.,?!\-'\"#&*^_]", "", sentence)  # remove punctuation

        sentence = tknzr.tokenize(sentence)
        sentence = [w for w in sentence if w not in unwanted]
        if stem:
            sentence = [st.stem(w) for w in sentence]

        df.set_value(i, "conversation", " ".join(sentence))


if __name__ == "__main__":
    df_path = "test_input.csv"

    df = pd.read_csv(df_path)
    sanitize(df, stem=True)

    save_path = df_path.replace(".csv", "_clean.csv")
    df.to_csv(save_path, index=False)
