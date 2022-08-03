import math

import nltk
import numpy as np
import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_lg")


class ExtractSyntacticFeat:
    """
    Syntactic features:
    ## POS distribution
    ## [] Number of subclauses in covering sentence
    ## [] Depth of constituent parse tree of the sentence
    ## [] Tense of main verb ()
    ## [] Modal verbs (boolean)
    """

    def __init__(self, doc):
        self.doc = doc
        syntactic_features = np.array([])

    def run(self):
        syntactic_features = np.array(self.pos_distribution() +
                                      self.count_subclauses() +
                                      self.depth_constituent_tree() +
                                      self.main_verb_tense() +
                                      self.contains_modal())
        return syntactic_features

    def pos_distribution(self):
        pos_distr = []
        tags = [token.tag_ for token in self.doc]
        for tag in nlp.pipe_labels['tagger']:
            pos_distr.append(math.log(1 + tags.count(tag)))
        # print("pos_distribution---",len(pos_distr))
        return pos_distr

    def count_subclauses(self):
        subclauses = 0
        for token in self.doc:
            if token.dep_ in ('xcomp', 'ccomp'):
                subclauses = subclauses + 1
        return [math.log(1 + subclauses)]

    def depth_constituent_tree(self):
        def tree_height(root):
            if not list(root.children):
                return 1
            else:
                return 1 + max(tree_height(x) for x in root.children)

        height = max([tree_height(token) for token in self.doc if token.dep_ == 'ROOT'])
        return [math.log(1 + height)]

    def main_verb_tense(self):
        verb_tags = ['VB', 'VBD', 'VBN', 'VBP', 'VBZ']
        main_verb_tense = [token.tag_ for token in self.doc if token.dep_ == 'ROOT']
        # print("main_verb_tense---",len([int(main_verb_tense[0]==tag) for tag in verb_tags]))
        # TODO watch out!! this should treated as categorical features!
        return [math.log(1 + int(main_verb_tense[0] == tag)) for tag in verb_tags]

    def contains_modal(self):
        # print("contains_modal---",len([int('MD' in [token.tag_ for token in doc])]))
        return [int('MD' in [token.tag_ for token in self.doc])]
