from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline


class OtherFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def other_features(self, input):
        ret = []
        for i in input:
            ret.append([1144])
            return ret

        def fit(self, X_input=None, y=None):
            self.other_features_list = self.other_features(X_input)
            self.model.fit(self.other_features_list, y)
            return self

        def transform(self, X_input=None):
            X_output = self.otherFeatures(X_input)
            X_output = self.model.transform(X_output)
            return X_output


def run_shit(X1):
    km = KMeans(n_clusters=5, n_init=5, init='k-means++')
    km_inside = KMeans(n_clusters=5, n_init=5, init='k-means++')
    feature_union = FeatureUnion(
        [('TfIdf', TfidfVectorizer(min_df=1, max_df=0.9, ngram_range=(1, 4), strip_accents='unicode', norm='l2')),
         ('OtherFeatures', OtherFeatures(km_inside))])

    feature_pipeline = Pipeline([('feature_union', feature_union), ("km", km)])
    km = feature_pipeline.fit(X1)
    return km
