from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import numpy as np
import pickle

from xgboost import XGBClassifier

classical_models = {
    'LogisticRegression': ("logreg", LogisticRegression),
    'MultinomialNB': ("mnb", MultinomialNB),
    'SVM': ("svm", SVC),
    'SGDClassifier': ("sgd", SGDClassifier),
    'RandomForest': ("random_forest", RandomForestClassifier),
    'XGB': ("xgboost", XGBClassifier),
    # 'Adaboost': ("adaboost", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    #                                            algorithm="SAMME",
    #                                            n_estimators=200))
}

_classical_models = {
    'LogisticRegression': ("logreg", LogisticRegression()),
    'MultinomialNB': ("mnb", MultinomialNB()),
    'SVM': ("svm", SVC(gamma='scale')),
    'SGDClassifier': ("sgd", SGDClassifier()),
    'RandomForest': ("random_forest", RandomForestClassifier()),
    'XGB': ("xgboost", XGBClassifier()),
    # 'Adaboost': ("adaboost", AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
    #                                            algorithm="SAMME",
    #                                            n_estimators=200))
}
selected_models = [
    'LogisticRegression', 'MultinomialNB', 'SVM', 'SGDClassifier', 'RandomForest', 'Adaboost'
]

vect_param_grid = {
    "tfidf__max_df": [.8],
    "tfidf__min_df": [.01, .005],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__norm": ['l2']
}

param_grid = {
    'LogisticRegression': {
        'logreg__C': [0.001, 0.01, 1, 10, 100],
        'logreg__max_iter': [100, 1000, 2000, 10000]
    },
    'MultinomialNB': {
        'mnb__alpha': np.linspace(0.5, 1.5, 4),
        'mnb__fit_prior': [True, False]
    },
    'SVM': {
        'svm__C': [10, 1.0, 0.1, 0.01],
        'svm__kernel': ['linear', 'poly']

    },
    'SGDClassifier': {
        'sgd__loss': ['log', 'hinge', 'modified_huber'],
        'sgd__penalty': ['l2', 'l1', 'elasticnet'],
        'sgd__alpha': [1e-3, 1e-2, 1e-1, 1, 5],
        'sgd__l1_ratio': [0.05, 0.1, 0.2, 1.0]
    },

    "RandomForest": {
        'random_forest__max_depth': [2]
    },
    "XGB":
        {
            'xgboost__n_estimators': [100, 250, 600],
            'xgboost__gamma': [0.1],
            'xgboost__learning_rate': [0.01, 0.1],
            'xgboost__max_depth': [10]
        },
    "Adaboost": {
        "adaboost__n_estimators": [100, 1000]
    }
}

simplified_grid = {
    'LogisticRegression': {
        'logreg__C': [1],
        'logreg__max_iter': [10000]
    }
}


# adapt: without tfdidf (already processed)
def run_multiple_classifiers(tag, classifier_list, X_tr, y_tr, X_tst=None, y_tst=None):
    classifiers = []
    predicted_ = []
    for model in classifier_list:
        try:
            print(f"training {model}")
            pipeline = Pipeline([
                # ('tfidf', TfidfVectorizer()),
                (classical_models[model])
            ])

            #            pipeline_param_grid = {**vect_param_grid, **param_grid[model]}
            pipeline_param_grid = {**simplified_grid[model]}

            grid_search = GridSearchCV(estimator=pipeline, param_grid=pipeline_param_grid, cv=5, n_jobs=-1)
            clf = grid_search.fit(X=X_tr, y=y_tr)
            classifiers.append(clf)

            # Save to file in the current working directory
            pkl_filename = f"../models/{tag}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(clf.best_estimator_, file)
            if X_tst and y_tst:
                print(clf.best_estimator_)
                predicted_labels = clf.predict(X_tst)
                predicted_.append(predicted_labels)
                print(f"results for model : {tag}::{model}")
                print(classification_report(y_true=y_tst, y_pred=predicted_labels))
            print("\n OK!")
        except ValueError as exc:
            print(exc)
    return classifiers, predicted_
