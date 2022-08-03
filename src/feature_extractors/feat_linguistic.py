from sklearn.feature_extraction.text import CountVectorizer

import spacy.tokens

# TODO: maybe

class ExtractLinguisticFeatures:
    def __init__(self):
        pass

    def rm_digits(self, text):
        # maybee relevant for Medical NER
        # text = re.sub(r'\d+', 'quantity', text)
        return text

    def _extract_frequent_unigrams(self, corpus):
        vectorizer = CountVectorizer(max_features=1000, binary=True, preprocessor=self.rm_digits())
        X = vectorizer.fit_transform(corpus)
        return X.toarray(), vectorizer
        # vectorizer_unigram_freq.get_feature_names()

    def pos_representation_feature(self, doc: spacy.tokens.Doc):
        transformed_text = ' '.join([token.tag_ for token in doc])
        return transformed_text

    def generate_dependency_tuples(doc):
        #doc = nlp(text)
        dependency_tuples = []
        for token in doc:
            dependency_tuples.append(token.text + '_' + token.head.text)
        return ' '.join(dependency_tuples)

    def extract_frequent_dependency_tuples(corpus, max_features):
        vectorizer = CountVectorizer(max_features=max_features, binary=True)
        X = vectorizer.fit_transform(corpus)
        return X.toarray()
