import re
import string
import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer


def sanitize(df, stem=False, lem=False):
    tknzr = TweetTokenizer()
    if stem:
        st = LancasterStemmer()
    if lem:
        lem = WordNetLemmatizer()
    stops = set(stopwords.words('english'))
    letters = set(string.ascii_lowercase)
    others = {"com", "www", "imgur", "org", "net", "id"}
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
        sentence = re.sub(r"[.,?!\-'\"#&*^_|]", "", sentence)  # remove punctuation

        sentence = tknzr.tokenize(sentence)

        if lem:
            sentence = [lem.lemmatize(w) for w in sentence]
        elif stem:
            sentence = [st.stem(w) for w in sentence]

        sentence = [w for w in sentence if w not in unwanted]

        df.set_value(i, "conversation", " ".join(sentence))


if __name__ == "__main__":
    df_path = "train_input1.csv"

    df = pd.read_csv(df_path)
    sanitize(df, lem=True)

    save_path = df_path.replace(".csv", "_clean.csv")
    df.to_csv(save_path, index=False)
