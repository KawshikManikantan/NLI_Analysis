import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from sklearn.linear_model import LogisticRegression
import re
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))


class BOWModel:

    stemmer = SnowballStemmer("english")

    def __init__(self, maxcost=20, all_features=False):
        self.maxcost = maxcost
        self.classifier = None
        self.all_features = all_features
    
    def getIDF(self, sentences):
        self.idf = {}
        for sentence in sentences:
            words = set(sentence.split())
            for word in words:
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        max_idf = 0
        self.MAX_IDF = ""
        for word in self.idf:
            self.idf[word] = np.log(len(sentences)/self.idf[word])
            if max_idf < self.idf[word]:
                max_idf = self.idf[word]
                self.MAX_IDF = word
    
    def weight(self, word):
        if word not in self.idf:
            return 1
        return min(1, self.idf[word]/self.idf[self.MAX_IDF])
    
    @staticmethod
    def edit_distance(w1, w2):
        len1 = len(w1)
        len2 = len(w2)
        dp = np.zeros((len1+1, len2+1), np.int32)
        for i in range(len1+1):
            dp[i, 0] = i
        for i in range(len2+1):
            dp[0, i] = i
        for i in range(1, len1+1):
            for j in range(1, len2+1):
                if w1[i-1] == w2[j-1]:
                    dp[i, j] = dp[i-1, j-1]
                else:
                    dp[i, j] = 1+min(dp[i-1, j-1], min(dp[i-1, j], dp[i, j-1]))
        return dp[len1, len2]

    def similiarity(self, w1, w2):
        stemmed_1 = BOWModel.stemmer.stem(w1)
        stemmed_2 = BOWModel.stemmer.stem(w2)
        if stemmed_1 == stemmed_2:
            return 1
        distance = BOWModel.edit_distance(stemmed_1, stemmed_2)
        return 1-distance/max(len(stemmed_1), len(stemmed_2))

    def wordAlignmentCost(self, w1, w2):
        sim = self.similiarity(w1, w2)
        if sim == 0:
            return self.maxcost
        cost = -np.log(sim)
        return min(self.maxcost, cost)

    def totalAlignmentCost(self, h, p):
        # h and p sentences
        h = h.split()
        p = list(set(p.split()))
        total_cost = 0
        costs = {}
        for i in range(len(h)):
            if h[i] not in costs:
                cost = self.maxcost
                for j in range(len(p)):
                    cost = min(cost, self.wordAlignmentCost(h[i], p[j]))
                costs[h[i]] = cost
            total_cost += self.weight(h[i]) * costs[h[i]]
        return total_cost
    
    def getFeatures(self, h, p):
        hCost = self.totalAlignmentCost(h, p)
        pCost = self.totalAlignmentCost(p, h)
        feqScore = np.exp(-hCost)
        reqScore = np.exp(-pCost)
        if self.all_features:
            eqvScore = feqScore * reqScore
            fwdScore = feqScore * (1-reqScore)
            revScore = (1-feqScore) * reqScore
            indScore = (1-feqScore) * (1-reqScore)
            return np.array([feqScore, reqScore, eqvScore, fwdScore, revScore, indScore])
        return np.array([feqScore, reqScore])

    def train(self, H, P, labels):
        assert len(P) == len(H) and len(H) == len(labels)
        combined = P.copy()
        combined.extend(H.copy())
        self.getIDF(combined)
        X = []
        y = []
        for i in range(len(labels)):
            if labels[i] == "entailment":
                y.append(0)
            elif labels[i] == "neutral":
                y.append(1)
            elif labels[i] == "contradiction":
                y.append(2)
            X.append(self.getFeatures(P[i], H[i]))
            if (i+1)%5000 == 0:
                print(i+1)
        X = np.array(X)
        y = np.array(y)
        self.classifier = LogisticRegression(penalty='l2', multi_class='ovr')
        self.classifier.fit(X, y)
    
    def predict(self, h, p):
        assert self.classifier is not None, "Please train the model first."
        result = self.classifier.predict(self.getFeatures(h, p).reshape(1, -1))
        if result == 0:
            return "entailment"
        elif result == 1:
            return "neutral"
        elif result == 2:
            return "contradiction"
        return "-"


def pre_process(text):
    text = text.lower()
    text = re.sub(r"`|~|!|@|#|\$|%|\^|&|\*|\(|\)|-|_|=|\+|\||\\|\[|\]|\{|\}|;|:|'|\"|,|<|>|\.|/|\?|\n|\t", " ", text)
    words = text.split()
    result = []
    for word in words:
        if word not in STOPWORDS:
            result.append(word)
    return " ".join(result)


def load_data(path, include_neutral=True):
    data = pd.read_csv(path, delimiter="\t")
    data = data[data["gold_label"] != "-"]
    if not include_neutral:
        data = data[data["gold_label"] != "neutral"]
    premise = data["sentence1"].astype(str).tolist()
    hypothesis = data["sentence2"].astype(str).tolist()
    labels = data["gold_label"].astype(str).tolist()
    return premise, hypothesis, labels


def main():
    import pickle
    snli = lambda type: f"../datasets/snli_1.0/snli_1.0_{type}.txt"
    p_train, h_train, train_labels = load_data(snli("train"))
    p_train = list(map(lambda x: pre_process(x), p_train))
    h_train = list(map(lambda x: pre_process(x), h_train))

    model = BOWModel()
    print("Training model")
    model.train(h_train, p_train, train_labels)
    print("Training completed")
    print("\n Saving the model")
    with open("bowModel.pkl", "wb") as modelFile:
        pickle.dump(model, modelFile)
    
    print("Testing the model")
    p_test, h_test, test_labels = load_data(snli("test"))
    p_test = list(map(lambda x: pre_process(x), p_test))
    h_test = list(map(lambda x: pre_process(x), h_test))

    correct = 0
    predictions = []
    label_types = ["entailment", "neutral", "contradiction"]
    confusion_matrix = {label_type: {label: 0 for label in label_types} for
                    label_type in label_types}
    for i in range(len(test_labels)):
        predictions.append(model.predict(h_test[i], p_test[i]))
        confusion_matrix[predictions[i]][test_labels[i]] += 1
        if predictions[-1] == test_labels[i]:
            correct += 1
    print("\nAccuracy:", correct/len(test_labels) * 100)
    metrics = ("precision", "Recall", "F1-Score", "Support")
    results = {label_type: {metric: 0 for metric in metrics} for label_type in label_types}
    for Type in label_types:
        TP = confusion_matrix[Type][Type]
        FP = 0
        FN = 0
        for other in label_types:
            if Type != other:
                FP += confusion_matrix[Type][other]
                FN += confusion_matrix[other][Type]
        results[Type][metrics[0]] = TP / (TP + FP)
        results[Type][metrics[1]] = TP / (TP + FN)
        results[Type][metrics[2]] = (2 * results[Type][metrics[0]] * results[Type][metrics[1]]) /\
                                    (results[Type][metrics[0]] + results[Type][metrics[1]])
        results[Type][metrics[3]] = 0
        for other in label_types:
            results[Type][metrics[3]] += confusion_matrix[other][Type]
    results_df = {}
    for metric in metrics:
        results_df[metric] = []
        for label_type in label_types:
            results_df[metric].append(results[label_type][metric])
    results_df = pd.DataFrame(results_df, index=label_types)
    results_df.to_csv("results.csv")
    print("Testing completed")
    print("Results saved")


if __name__ == "__main__":
    main()
