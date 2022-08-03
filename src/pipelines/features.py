import ast
import math
from typing import List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from feature_extractors.feat_indicators import ExtractDiscourseIndicators
from feature_extractors.feat_sentiment import SentimentClassifier
from feature_extractors.feat_syntactic import ExtractSyntacticFeat


class FeatureExtractorMethod:
    pass


def transformer_list_from_feature_config(feature_config: dict):
    for feature_extractor in list(feature_config.keys()):
        return (feature_extractor, FeatureExtractorMethod)


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""
    sentence_markers = ['.,;?!:\n\r']

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [{'length': math.log(1 + len(text)),
                 'num_sentences': math.log(1 + sum([text.count(sign) for sign in self.sentence_markers]))}
                for text in texts]


class SpacyVectors(BaseEstimator, TransformerMixin):
    """ Get vector for sentence """

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [text.vector for text in texts]


class DocumentEmbeddings(BaseEstimator, TransformerMixin):
    """Generate a document embedding"""

    def __init__(self, key: str):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # self.X_train['doc2vec'], self.X_test['doc2vec'] = doc2vec_(X_train=self.X_train['sentences'],
        # X_test=self.X_test['sentences'])
        data = [vect for vect in data_dict[self.key]]
        return data


# class WordEmbeddings(BaseEstimator, TransformerMixin):
#     """ Get vector for sentence """
#
#     def fit(self, x, y=None):
#         return self
#
#     def transform(self, texts):
#         get_vector = ExtractEmbeddingVectors().calculate_sum_vectors
#         return [get_vector(text) for text in texts]


class SentimentExtractor_(BaseEstimator, TransformerMixin):
    """ Loads pregenerated sentiment categorization"""

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        texts = list(texts)
        preds = SentimentClassifier().predict(texts)
        return preds


class SentimentExtractor(BaseEstimator, TransformerMixin):
    """ Loads pregenerated sentiment categorization"""
    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # TODO watch out! Logarithm is used to standardize the values (counters)
        data = [[math.log(1 + value) for value in vect] for vect in data_dict['sentiment']]
        return data

class FoundEntities(BaseEstimator, TransformerMixin):
    """
    Using spaCy'S default entities
    # Labels:
    #    CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY,
    #    NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
    """

    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        labels = ["CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY", "NORP", "ORDINAL",
                  "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART"]

        def count_entities(doc):
            count = {}
            ents = [ent.label_ for ent in doc.ents]
            for label in labels:
                count[label] = math.log(1 + ents.count(label))
            return count

        return [count_entities(doc) for doc in docs]


class Initialiser(BaseEstimator, TransformerMixin):

    def __init__(self, super_self):
        self.df = super_self

    def fit(self, x, y=None):
        return self

    def transform(self, x, super_self):
        if super_self:
            super_self.df = x
            return super_self
        else:
            return x


class DiscourseIndicators(BaseEstimator, TransformerMixin):
    """ Get vector for sentence """

    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        get_indicators = ExtractDiscourseIndicators().indicator_features
        return [get_indicators(text) for text in texts]


class SyntacticFeatures(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        return [ExtractSyntacticFeat(doc=doc).run() for doc in docs]


class MedicalFeatures(BaseEstimator, TransformerMixin):
    """ Get vector for sentence """

    def __init__(self, **kwargs):
        if 'use_this_file' in kwargs:
            pass
        elif 'call_service' in kwargs:
            pass
        elif 'use_column' in kwargs:
            self.use_column = kwargs['use_column']
        else:
            pass

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        # TODO watch out! Logarithm is used to standardize the values (counters)
        data = [[math.log(1 + value) for value in ast.literal_eval(vect)] for vect in data_dict[self.use_column]]
        return data


class ContextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, super_self, use_context=False):
        print("Super_self", type(self))
        self.super_self = super_self
        self.use_context = use_context

    def fit(self, x, y=None):
        return self

    def transform(self, features):
        if not self.use_context:
            return features

        if features.shape[0] == self.super_self.X_train.shape[0]:
            df = self.super_self.X_train
        else:
            df = self.super_self.X_test
        if not isinstance(features, np.ndarray):  # if csr matrix, or?
            features = features.todense()
        df['new_features'] = features.tolist()
        df['new_features'] = df['new_features'].apply(np.array)
        # del _
        context = df.apply(lambda row: self.get_context_features(df, row, 2), axis=1).to_list()

        return csr_matrix(np.stack(context))

    def get_context_features(self, df, row, window: int = 1):
        doc_id = row['doc_id']
        df_ = df.query(f'doc_id == "{doc_id}"').reset_index()
        pos_in_comment = list(df_['Unnamed: 0.1.1.1']).index(row['Unnamed: 0.1.1.1'])
        n_features = df_['new_features'].iloc[0].shape[0]
        before = pos_in_comment - window
        if before < 0:
            before = abs(before)
            previous = np.zeros(before * n_features)
            if window - before != 0:
                previous = np.append(previous, np.concatenate(
                    df_[pos_in_comment - (window - before):pos_in_comment]['new_features'].tolist()))
        else:
            previous = np.concatenate(df_[pos_in_comment - window:pos_in_comment]['new_features'].tolist())
        output = np.append(previous, row['new_features'])

        # how many positions would be exceeded after considering the window?
        after = (len(df_) - 1) - (pos_in_comment + 2)
        if after < 0:
            after = abs(after)
            next = np.zeros(after * n_features)
            if window - after != 0:
                next = np.append(next,
                                 np.concatenate(df_[pos_in_comment + 1:pos_in_comment + window - after + 1][
                                                    'new_features'].tolist()))
        else:
            next = np.concatenate(df_[pos_in_comment + 1:pos_in_comment + window + 1]['new_features'].tolist())

        output = np.append(output, next)
        return output
