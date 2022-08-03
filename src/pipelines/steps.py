from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from feature_extractors.feat_embeddings import Doc2VecTransformer
from pipelines.features import TextStats, DiscourseIndicators, SyntacticFeatures, SentimentExtractor, \
    DocumentEmbeddings, MedicalFeatures, FoundEntities, ItemSelector

features_to_unite = [
    # ('init', Pipeline([('init', Initialiser(super_self=self))])),
    # Pipeline for pulling features from the post's subject line
    ('tfidf', Pipeline([
        ('selector', ItemSelector(key='tfidf_ready_sentence')),
        ('tfidf', TfidfVectorizer()),
        # ('svd', TruncatedSVD(50)),
    ])),
    # Pipeline for pulling ad hoc features from post's body
    ('body_stats', Pipeline([
        ('selector', ItemSelector(key='sentences')),
        ('stats', TextStats()),  # returns a list of dicts
        ('vect', DictVectorizer()),  # list of dicts -> feature matrix
        ('standardscaler', StandardScaler(with_mean=False)),
        ('normalizer', Normalizer()),
    ])),
    ('indicators', Pipeline([
        ('selector', ItemSelector(key='sentences')),
        ('indicators', DiscourseIndicators()),
        ('standardscaler', StandardScaler(with_mean=False)),
        ('normalizer', Normalizer()),
        # ('svd', TruncatedSVD(10)),
    ])),
    ('syntactic', Pipeline([
        ('selector', ItemSelector(key='doc')),
        ('syntactic', SyntacticFeatures()),
        ('standardscaler', StandardScaler(with_mean=False)),
        ('normalizer', Normalizer())
    ])),
    ('default_entities', Pipeline([
        ('selector', ItemSelector(key='doc')),
        ('indicators', FoundEntities()),
        ('vect', DictVectorizer()),  # list of dicts -> feature matrix
        ('standardscaler', StandardScaler(with_mean=False)),
        ('normalizer', Normalizer()),
        # ('svd', TruncatedSVD(10)),
    ])),
    ('embeddings', Pipeline([
        # ('selector', ItemSelector(key='tfidf_ready_sentence')),
        # ('embedding', DocumentEmbeddings(key='doc2vec')),
        ('embeddings', Doc2VecTransformer(field='sentences', vector_size=300)),
        # ('embedding', DocEmbeddings()),
        # ('embedding', WordEmbeddings()),
        # ('standardscaler', StandardScaler()),
        ('pca', TruncatedSVD(50)),
    ])),
    ('medical', Pipeline([
        ('medical_entities', MedicalFeatures(use_column='metamap_feats')),
        ('standardscaler', StandardScaler(with_mean=False)),
        ('normalizer', Normalizer()),
        ('svd', TruncatedSVD(50)),
    ])),
    ('sentiment', Pipeline([
        #('selector', ItemSelector(key='sentences')),
        ('sentiment_class', SentimentExtractor())
    ])),
]
