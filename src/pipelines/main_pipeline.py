import ast
import json
import pickle
from abc import ABC
from statistics import mode

import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as LE

from classifiers.classifiers import run_multiple_classifiers

data_path = '../data'


class FeatureExtraction:
    steps_by_files = ['input',
                      'df_all__featured_tfm-pipeline_v1',
                      'df_lab__featured+metamap_v1.csv',
                      'df_lab__extended_features'  # with embeddings
                      ]


class Data:
    features = {
        'merged_features': [
            'feat_freq_unigrams',
            'feat_freq_dep_tupl',
            'feat_synt_and_indic',
            ''

        ],
        'merged_features_extended': ['feat_pos_rep'],
        'metamap_feats': []

    }

    def __init__(self, input_data):
        self.input_data = pd.read_csv(input_data).sample(n=1000, random_state=54)


class Pipeline:
    allowed_labels = ['agreed_labels', 'binary_labels']
    default_classifiers = ['LogisticRegression']

    def __init__(self, input_data: str, sampling=False):
        if sampling:
            drugArg = pd.read_csv(input_data)  # .sample(n=1000, random_state=54)
        else:
            drugArg = pd.read_csv(input_data)
        drugArg['feats'] = drugArg['feats'].apply(lambda x: json.loads(x))
        drugArg['binary_labels'] = drugArg['agreed_labels'].apply(
            lambda label: 'RELATED' if label != 'NON_RELATED' else label)

        self.input_data = drugArg
        del drugArg

    def do_run(self):
        print(self.input_data['agreed_labels'].value_counts())
        # print("1----", self.input_data[self.input_data.index.isin([120])])

    def split(self):
        le = LE()
        le.fit(['NON_RELATED', 'RELATED', 'supporting', 'attacking'])

        X = self.input_data.feats  # .apply(lambda x: [float(elem) for elem in x]).tolist()
        y = self.input_data.agreed_labels
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=54, stratify=y)
        print(f"Train dataset size: {len(y_train)} \n Test dataset size: {len(y_test)}")

        return X_train, X_test, y_train, y_test, le

    def train(self, label: str = None):
        X_train, X_test, y_train, y_test, le = self.split()

        train_indexes = X_train.index
        test_indexes = X_test.index
        y_train = le.transform(list(y_train))
        y_test = le.transform(list(y_test))

        train_dataset = self.input_data[self.input_data.index.isin(list(train_indexes))]
        test_dataset = self.input_data[self.input_data.index.isin(list(test_indexes))]

        y_train_binary = le.transform(list(train_dataset['binary_labels']))
        y_test_binary = le.transform(list(test_dataset['binary_labels']))

        run_multiple_classifiers("MAIN", self.default_classifiers, X_train.to_list(), y_train_binary, y_test_binary)

        train_attacking = train_dataset.query('agreed_labels != "supporting"')
        train_supporting = train_dataset.query('agreed_labels != "attacking"')

        X_att = train_attacking.feats.apply(lambda x: [float(elem) for elem in x])
        y_att = train_attacking.agreed_labels
        y_att = le.transform(list(y_att))

        run_multiple_classifiers("ATTACK", self.default_classifiers, X_att.to_list(), y_att)

        X_supp = train_supporting.feats.apply(
            lambda x: [float(elem) for elem in x])
        y_supp = train_supporting.agreed_labels
        y_supp = le.transform(list(y_supp))
        run_multiple_classifiers("SUPPORT", self.default_classifiers, X_supp.to_list(), y_supp)

        return X_test, y_test

    def run(self):
        X_train, X_test, y_train, y_test, le = self.split()
        print(le.classes_)
        y_expected = le.transform(list(y_test))
        y_trainable = le.transform(list(y_train))
        y_obtained = {}
        y_post_train = {}
        models = {}
        for classifier in ['MAIN', 'SUPPORT', 'ATTACK']:
            with open(f"../models/{classifier}.pkl", "rb") as file:
                models[classifier] = pickle.load(file)

        for classifier in ['MAIN', 'SUPPORT', 'ATTACK']:
            y_obtained[classifier] = models[classifier].predict(X_test.to_list())
            y_post_train[classifier] = models[classifier].predict(X_train.to_list())
            print(f"predicting for {models[classifier].classes_}")
            y_obtained[f"{classifier}_prob"] = np.amax(models[classifier].predict_proba(X_test.to_list()), axis=1)
            y_post_train[f"{classifier}_prob"] = np.amax(models[classifier].predict_proba(X_train.to_list()), axis=1)

        # output_layer
        output_clf = self.create_output_layer(y_post_train, y_trainable)

        #####

        print(y_obtained)
        df = pd.DataFrame(y_obtained)
        df['expected'] = y_expected
        df['vote'] = df.apply(lambda row: vote(row), axis=1)
        df['vote_prob'] = df.apply(lambda row: vote_conf(row), axis=1)
        test_indexes = X_test.index
        df['index'] = test_indexes
        df['text'] = self.input_data[self.input_data.index.isin(list(test_indexes))]['sentence']
        df['review'] = self.input_data[self.input_data.index.isin(list(test_indexes))]['review']
        df['feats'] = df.apply(lambda row: generate_feats(row), axis=1)
        df['output_layer'] = output_clf.predict(df.feats.to_list())

        df['meta_vote'] = df.apply(lambda row: majority_vote(row), axis=1)

        df.to_csv("results_informed.csv")
        print("vote", classification_report(y_true=df['expected'], y_pred=df['vote']))
        print("output_layer", classification_report(y_true=df['expected'], y_pred=df['output_layer']))
        print("meta_vote", classification_report(y_true=df['expected'], y_pred=df['meta_vote']))

    # return y_obtained

    def create_output_layer(self, y_post_train, y_expected):
        df_output_layer = pd.DataFrame(y_post_train)
        df_output_layer['expected'] = y_expected
        df_output_layer['feats'] = df_output_layer.apply(lambda row: generate_feats(row), axis=1)
        X = df_output_layer.feats
        y = df_output_layer.expected
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=54, stratify=y)
        clf = RandomForestClassifier(max_depth=20, random_state=0).fit(X_train.to_list(), y_train.to_list())
        predicted_labels = clf.predict(X_test.to_list())
        print("output layer: \n", classification_report(y_true=y_test, y_pred=predicted_labels))
        df_output_layer.to_csv("output_layer_creation.csv")
        return clf


def majority_vote(row):
    return mode([row['vote'], row['vote_prob'], row['output_layer']])


def generate_feats(row):
    feats = []
    for col in ['MAIN', 'SUPPORT', 'ATTACK']:
        feats.append(row[col])
        feats.append(row[f"{col}_prob"])
    return feats


def vote_conf(row):
    probs = {col: row[f"{col}_prob"] for col in ['MAIN', 'SUPPORT', 'ATTACK']}
    best_label = max(probs, key=probs.get)
    return int(row[best_label])


def vote_informed(row):
    # base
    vote = vote_conf(row)
    # corrections
    if row['MAIN'] == row['SUPPORT'] == row['ATTACK'] == 0:
        return 0
    if row['MAIN'] == 0 and row['SUPPORT'] == 3 and row['ATTACK'] == 2:
        return random.choice([2, 3])  # Use probabilities here
    if row['MAIN'] == 0 and row['SUPPORT'] == 0 and row['ATTACK'] == 2:
        return 2
    if row['MAIN'] == 0 and row['SUPPORT'] == 3 and row['ATTACK'] == 0:
        return 3
    if row['MAIN'] == 1 and (row['SUPPORT'] == 0 or row['ATTACK'] == 0):
        return 0
    if vote == 3 and row['SUPPORT_prob'] <= row['MAIN_prob'] and row['SUPPORT_prob'] <= row['ATTACK_prob']:
        return 0
    if vote == 0 and row['MAIN_prob'] <= 0.9 and row['MAIN_prob'] and row['SUPPORT_prob'] <= row['ATTACK_prob']:
        return 3
    else:
        return vote


def vote(row):
    threshold = 0.6

    # empirical

    if row['MAIN'] == 0 and row['SUPPORT'] == 3 and row['ATTACK'] == 0:
        return 3
    if row['MAIN'] == 0 and row['SUPPORT'] == 0 and row['ATTACK'] == 2:
        return 2
    if row['MAIN'] == 0 and row['SUPPORT'] == 3 and row['ATTACK'] == 2:
        return 0
    # if row['MAIN'] == 1 and (row['SUPPORT'] == 0 or row['ATTACK'] == 0):
    #     return 0
    # if row['MAIN'] == 1 and row['SUPPORT'] == 3:
    #     return 3
    # if row['MAIN'] == 1 and row['ATTACK'] == 2:
    #     return 2
    ######
    # a priori heuristics
    if row['MAIN'] == 1:
        if row['SUPPORT'] == 3:
            if row['ATTACK'] == 2:
                if row['SUPPORT_prob'] >= row['ATTACK_prob']:  # Here the highest score
                    return 3
                else:
                    return 2
            if row['ATTACK'] == 0:
                return 3
            else:
                raise Exception("not expected")
        if row['SUPPORT'] == 0:
            if row['ATTACK'] == 2:
                return 2
            else:
                return 0
    if row['MAIN'] == 0:
        if row['SUPPORT'] == 3:
            if row['MAIN_prob'] > row['SUPPORT_prob'] and row['MAIN_prob'] > row['ATTACK_prob']:
                return 0
            if row['ATTACK'] == 2:
                if row['SUPPORT_prob'] >= row['ATTACK_prob']:  # Here the highest score
                    return 3
                else:
                    return 2
            else:
                return 0
        else:
            return 0


if __name__ == "__main__":
    input_data = f'{data_path}/df_lab__feat_matrix_1.csv'
    pipe = Pipeline(input_data=input_data)
    pipe.do_run()
    X_test, y_test = pipe.train()
    print(y_test)
