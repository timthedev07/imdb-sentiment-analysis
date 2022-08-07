import os
import string
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.keras import Sequential as ST
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.layers import TextVectorization
import nltk
import numpy as np
from nltk.corpus import stopwords

pj = os.path.join

SAVED_MODEL_DIR = "model"

# the number of nodes in the hidden layers
EPOCHS = 15
VOCAB_SIZE = 10000
SEQUENCE_LENGTH = 100

def custom_standardization(input_data):
    stop_words = set(stopwords.words('english'))

    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_html = tf.strings.regex_replace(stripped_html,r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', ' ')
    stripped_html = tf.strings.regex_replace(stripped_html, r'@([A-Za-z0-9_]+)', ' ' )
    for i in stop_words:
        stripped_html = tf.strings.regex_replace(stripped_html, f' {i} ', " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


def readCSV(dataDir = "data-csv"):
    train = pd.read_csv(pj(dataDir, "train.csv"), encoding="utf-8")
    test = pd.read_csv(pj(dataDir, "test.csv"), encoding="utf-8")
    return (train, test)

def getDataset(dataSource="small"):
    if dataSource == "small":
        (train, test) = readCSV()
    else:
        (train, test) = readCSV("large-data-csv")

    trainX = train.loc[:, "review"].values
    trainY = train.loc[:, "sentiment"].values

    testX = test.loc[:, "review"].values
    testY = test.loc[:, "sentiment"].values

    return (trainX, trainY, testX, testY)

def getTrainedModel() -> ST:
    (trainX, trainY, testX, testY) = getDataset(dataSource="large")
    print(testY.head)

    # Vocabulary size and number of words in a sequence.

    nltk.download('stopwords')

    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=SEQUENCE_LENGTH)

    vectorize_layer.adapt(np.concatenate([trainX, testX]))

    embedding_dim = 32

    model = Sequential([
        vectorize_layer,
        Embedding(VOCAB_SIZE, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(trainX, trainY, epochs=EPOCHS, batch_size=32)
    print("\nTesting result:")
    model.evaluate(testX, testY, verbose=2)

    model.save("model")

    return model

def loadModel() -> ST:
    custom_objects = {"custom_standardization": custom_standardization}
    with tf.keras.utils.custom_object_scope(custom_objects):
        loaded = tf.keras.models.load_model(SAVED_MODEL_DIR)
        return loaded

def main():
    model = loadModel()

    # the model outputs a negative value for a negative sentiment, and vice versa
    with open("sample.txt", "r", encoding="utf-8") as f:
        review = f.read()
        [[res]] = model.predict([review])
        print(res)
        print(f"Concluded sentiment: {'negative' if res < 0 else ('positive' if res > 0 else 'neutral')}")

if __name__ == "__main__":
    main()
