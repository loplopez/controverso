import ast
import math
import time
from typing import Tuple

import numpy as np
import pandas as pd
import spacy
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, PCA

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder as LE
from xgboost import XGBClassifier

from classifiers.classifiers import classical_models
from feature_extractors.feat_embeddings import doc2vec_
from feature_extractors.feat_sentiment import SentimentClassifier
from pipelines import steps
from pipelines.features import ContextFeatures

from pipelines.utils import grouped_train_test_split
from preprocessors.preprocess import decontract, extract_tokens

le = LE()
nlp = spacy.load("en_core_web_lg")


class FeatureExtractor:
    def __init__(self, input_path: str = None, binary: bool = False, use_context: bool = False,
                 config_name: str = "default",
                 sample_size: int = None, sentiment: bool = False, from_path: str = None):
        self.y_test = None
        self.y_train = None
        self.config_name = config_name
        if from_path:
            self.X_train = pd.read_pickle(f"{from_path}_train.pkl")
            self.X_test = pd.read_pickle(f"{from_path}_test.pkl")
        else:
            # TODO adapt by-demand: what is done during pre-processing and what's during feature extraction
            cols = ['Unnamed: 0.1.1.1', 'use_case', 'doc_id', 'n_sentences', 'sentences', 'metamap_feats',
                    'agreed_labels']
            if sample_size:
                self.input_data = pd.read_csv(input_path)[cols].sample(n=sample_size)
            else:
                self.input_data = pd.read_csv(input_path)[cols]
            print("Decontracting expressions...")
            self.input_data['sentences'] = self.input_data['sentences'].apply(lambda str: decontract(str))
            print("Extracting NLP properties...")
            self.input_data['doc'] = self.input_data['sentences'].apply(nlp)
            print("Extracting tokens...")
            self.input_data['tfidf_ready_sentence'] = self.input_data['doc'].apply(extract_tokens)

            # le.fit(list(self.input_data['target'].unique())

            if sentiment:
                print("Processing sentiment ...")
                tic = time.time()
                self.input_data['sentiment'] = SentimentClassifier().predict(self.input_data['sentences'].to_list())
                print(f"elapsed_time with sentiment analysis {time.time() - tic}")

    def split_train_test(self, test_size: float = 0.2, random_state: int = 7, as_binary: bool = False,
                         persist: bool = False):
        if as_binary:
            self.input_data['binary_agreed_labels'] = self.input_data['agreed_labels'].apply(
                lambda label: 'RELATED' if label != 'NON_RELATED' else label)
            self.input_data['target'] = self.input_data['binary_agreed_labels']
        else:
            self.input_data['target'] = self.input_data['agreed_labels']
        self.input_data['target'] = le.fit_transform(self.input_data['target'].to_list())
        #################################################################
        self.X_train, self.X_test, self.y_train, self.y_test = grouped_train_test_split(
            self.input_data[
                ['Unnamed: 0.1.1.1', 'use_case', 'doc_id', 'n_sentences', 'sentences', 'doc', 'tfidf_ready_sentence',
                 'metamap_feats', 'target', 'sentiment', 'agreed_labels']],  # binary_agreed_labels ?? TODO: fix
            self.input_data.target, group_by='doc_id', test_size=test_size,
            random_state=random_state)

        print(self.y_train.value_counts(), self.y_test.value_counts())
        if persist:
            self.X_train.to_pickle(f'preproc_data__{self.config_name}_train.pkl')
            self.X_test.to_pickle(f'preproc_data__{self.config_name}_test.pkl')

    def extract_vectors(self):
        return doc2vec_(self.input_data['sentences'])

    def run_feature_extraction(self, feature_config: dict, context: bool = False):
        if not feature_config:
            transformer_weights = {
                                      # 'init': 0,
                                      'tfidf': 1,
                                      'indicators': 1,
                                      'body_stats': 1,
                                      'embeddings': 1,
                                      'default_entities': 1,
                                      'medical': 1,
                                      'syntactic': 1,
                                      'sentiment': 1,
                                  },
        pipeline = Pipeline([
            # Use FeatureUnion to combine the features from subject and body
            ('union', FeatureUnion(
                transformer_list=[extractor for extractor in steps.features_to_unite if
                                  extractor[0] in list(feature_config.keys())],
                # weight components in FeatureUnion
                transformer_weights=feature_config,
            )),
            ('context', Pipeline([
                ('context', ContextFeatures(super_self=self, use_context=context)),
                ('standardscaler', StandardScaler(with_mean=False)),
                # ('svd', TruncatedSVD(50)),
                ('normalizer', Normalizer())
            ])),
        ])

        return pipeline

    def run_classification(self, model: str, features: dict, num_classes: int = None, feature_config: dict = None,
                           context: bool = None):
        classify_pipe = Pipeline([
            # ('svd', TruncatedSVD(n_components=30)),
            # Use a SVC classifier on the combined features
            load_model_config(model_id=model, num_classes=num_classes),
        ])

        X_train = features['train']
        X_test = features['test']
        _ = classify_pipe.fit(X_train, self.y_train)
        y = classify_pipe.predict(X_test)
        print(classification_report(y, self.y_test))
        report = classification_report(y, self.y_test, output_dict=True)
        pd.DataFrame(report).T.to_csv(f'config_{self.config_name}.csv')

        return report, y


class PersistFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, super_self):
        self.super_self = super_self

    def fit(self, x, y=None):
        return self

    def transform(self, features):
        if features.shape[0] == self.super_self.X_train.shape[0]:
            set_name = "train"
        else:
            set_name = "test"
        try:
            sparse.save_npz(f"{self.super_self.config_name}___features__{set_name}.npz", features)
        except AttributeError:
            with open(f"{self.super_self.config_name}___features__{set_name}.npy", 'wb') as f:
                np.save(f, features)

        return features


def load_model_config(model_id: str, num_classes: int = None) -> Tuple:
    model = classical_models[model_id]
    print(type(model[1]))
    if model_id == 'XGB':
        return (model_id, model[1](num_classes=num_classes))
    else:
        return (model_id, model[1]())
