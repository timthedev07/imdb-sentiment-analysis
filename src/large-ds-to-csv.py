import os
import pandas as pd

dataDir = "data"
csvDir = "large-data-csv"

TRAIN_PERCENTAGE = 7.5/10

labels = {
    "positive": 1,
    "negative": 0
}

def readDataToDF():
    """
    This returns a tuple, (train, test), with two pandas data frames;
    one contains training data while the other contains testing data
    """

    allData = pd.read_csv(os.path.join(dataDir, "large.csv"))

    # replacing words with 1/0
    allData.loc[allData["sentiment"] == "negative", "sentiment"] = 0
    allData.loc[allData["sentiment"] == "positive", "sentiment"] = 1

    n = len(allData.index)
    split = int(n * TRAIN_PERCENTAGE)

    train = allData.iloc[:split]

    test = allData.iloc[split:]

    return (train, test)

def main():
    (train, test) = readDataToDF()
    train.to_csv(os.path.join(csvDir, "train.csv"), index=False, encoding="utf8")
    test.to_csv(os.path.join(csvDir, "test.csv"), index=False, encoding="utf8")


if __name__ == "__main__":
    main()