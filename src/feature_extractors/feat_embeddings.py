import multiprocessing
import sys

import numpy as np

import gensim
import gensim.downloader as gensim_api
import pandas as pd
from gensim.models import Doc2Vec, doc2vec
from gensim.models import Word2Vec

# nlp = gensim_api.load("word2vec-google-news-300")
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing import preprocess_string
from sklearn import utils
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

# wv = gensim_api.load("glove-wiki-gigaword-300")
# wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
dtf_train = pd.DataFrame()


class ExtractEmbeddingVectors:
    # currently using gloVe
    file_path = "../../TFM_artifacts/EN-wform.w.5.cbow.neg10.400.subsmpl.txt"

    def __init__(self, dimensions: int = 400):
        self.dimensions = dimensions
        print("Loading Glove Model")
        f = open(self.file_path, 'r', encoding="utf-8")
        model = {}
        for i, line in enumerate(f):
            splitLine = line.split()
            word = splitLine[0]
            # check correct dimensions
            if len(splitLine[1:]) == self.dimensions:
                try:
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    model[word] = embedding
                except:
                    print(sys.exc_info())
                    print("error for word", word)
                    print("error in line ", str(i))
            else:
                print("Incorrect dimensions: word", word, "not included")
        print("Done!", len(model), " words loaded!")
        self.loaded_vectors = model

    def calculate_sum_vectors(self, doc: str):
        # doc = nlp(text)
        # token_array = [word.text for word in doc]
        token_array = doc
        result = np.zeros(self.dimensions)
        for token in token_array:
            if token in self.loaded_vectors:
                result = self.loaded_vectors[token]
        return result


#
# class ExtractEmbeddings:
#     corpus = dtf_train["text_clean"]
#
#     ## create list of lists of unigrams
#     lst_corpus = []
#     for string in corpus:
#         lst_words = string.split()
#         lst_grams = [" ".join(lst_words[i:i + 1])
#                      for i in range(0, len(lst_words), 1)]
#         lst_corpus.append(lst_grams)
#
#     nlp = gensim.models.word2vec.Word2Vec(lst_corpus, size=300,
#                                           window=8, min_count=1, sg=1, iter=30)
#
#
# def word_averaging(wv, words):
#     all_words, mean = set(), []
#
#     for word in words:
#         if isinstance(word, np.ndarray):
#             mean.append(word)
#         elif word in wv.vocab:
#             mean.append(wv.syn0norm[wv.vocab[word].index])
#             all_words.add(wv.vocab[word].index)
#
#     if not mean:
#         #        logging.warning("cannot compute similarity with no input %s", words)
#         print("!!!")
#         # FIXME: remove these examples in pre-processing
#         return np.zeros(wv.layer1_size, )
#
#     mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
#     return mean
#
#
# def word_averaging_list(wv, text_list):
#     return np.vstack([word_averaging(wv, review) for review in text_list])


# def w2v_tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text, language='english'):
#         for word in nltk.word_tokenize(sent, language='english'):
#             if len(word) < 2:
#                 continue
#             tokens.append(word)
#     return tokens

#
# def test_embeddings():
#     # wv = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
#     wv.init_sims(replace=True)
#
#     def word_averaging(wv, words):
#         all_words, mean = set(), []
#         for word in words:
#             if isinstance(word, np.ndarray):
#                 mean.append(word)
#             elif word in wv.vocab:
#                 mean.append(wv.syn0norm[wv.vocab[word].index])
#                 all_words.add(wv.vocab[word].index)
#         if not mean:
#             # logging.warning("cannot compute similarity with no input %s", words)
#             # FIXME: remove these examples in pre-processing
#             return np.zeros(wv.vector_size, )
#         mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
#         return mean
#
#     def word_averaging_list(wv, text_list):
#         return np.vstack([word_averaging(wv, post) for post in text_list])


def doc2vec_(X_train, X_test):
    # TODO: simplify this shit
    def label_sentences(corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
        return labeled

    X_train = label_sentences(X_train, 'Train')
    X_test = label_sentences(X_test, 'Test')

    all_data = X_train + X_test

    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.dv[prefix]
        return vectors

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

    print(f"Dimensions of vectors:: train, test:: {len(train_vectors_dbow)}, {len(test_vectors_dbow)}")

    return train_vectors_dbow.tolist(), test_vectors_dbow.tolist()


class Doc2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, field: str, vector_size=300, learning_rate=0.02, epochs=30):
        self.field = field
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count()

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(preprocess_string(row[self.field]), [index]) for index, row in
                    df_x.iterrows()]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)
        for epoch in range(self.epochs):
            model.train(utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha
        self._model = model
        return self

    def transform(self, df_x):
        return np.asmatrix(np.array(
            [self._model.infer_vector(preprocess_string(row[self.field])) for index, row in df_x.iterrows()]))
