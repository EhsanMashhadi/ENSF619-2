from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


class Preprocessing:
    def split(self, df):
        x = df["text"]
        y = df["label"]
        train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=1, test_size=0.2)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.1)
        print(train_x.describe)
        print(train_y.describe)

        return {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y, "test_x": test_x,
                "test_y": test_y}

    def split_vectorized(self, vectors, target):
        train_x, test_x, train_y, test_y = train_test_split(vectors, target, random_state=1, test_size=0.2)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y, test_size=0.1)
        return {"train_x": train_x, "train_y": train_y, "val_x": val_x, "val_y": val_y, "test_x": test_x,
                "test_y": test_y}

    def change_column(self, df):
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        df.rename(columns={'v1': 'label', 'v2': 'text'}, inplace=True)
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    def word_to_vector(self, df):
        tf_ifd = TfidfVectorizer()
        vectors = tf_ifd.fit_transform(df["text"])
        print("Vector Size: {}".format(vectors.shape))
        return vectors

    def count_tokens(self, df):
        sentences = df["text"]
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
        max_len = 0
        for sentence in sentences:
            max_len = max(len(tokenizer.encode(sentence, add_special_tokens=True)), max_len)
        return max_len
