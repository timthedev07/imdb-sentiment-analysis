import os
import pandas as pd

dataDir = "data/aclImdb"
csvDir = "data-csv"

labels = {
    "pos": 1,
    "neg": 0
}

def readDataToDF():
    """
    This returns a tuple, (train, test), with two pandas data frames;
    one contains training data while the other contains testing data
    """

    dfs = {
        "train": pd.DataFrame(),
        "test": pd.DataFrame()
    }

    cols = ["review", "sentiment"]

    for dataCategory in ["train", "test"]:
        for label in labels.keys():
            currLabelDir = os.path.join(dataDir, dataCategory, label)

            for reviewTxtFname in os.listdir(currLabelDir):
                with open(os.path.join(currLabelDir, reviewTxtFname), "r", encoding='utf-8') as f:
                    raw = f.read().strip("\n")
                    dfs[dataCategory] = dfs[dataCategory].append([[raw, labels[label]]], ignore_index=True)

    dfs["train"].columns = cols
    dfs["test"].columns = cols

    return (dfs["train"], dfs["test"])

def main():
    (train, test) = readDataToDF()
    train.to_csv(os.path.join(csvDir, "train.csv"), index=False, encoding="utf8")
    test.to_csv(os.path.join(csvDir, "test.csv"), index=False, encoding="utf8")


if __name__ == "__main__":
    main()