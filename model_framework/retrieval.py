import pandas as pd
import logging
import random
import numpy as np
import re
import os
from framework import Task, ModelTrainer, FeatureEngineering, Evaluation, DataCuration
from rank_bm25 import BM25Okapi
import spacy
import en_core_web_sm
NLP = en_core_web_sm.load(disable=["tagger", "ner"])

class TaskRetrieval(Task):
    def __init__(self, config):
        self.config = config
    

class FeatureEngineeringRetrieval(FeatureEngineering):
    def __init__(self, data_args):
        self.data = data_args['dataset']
        self.data_args = data_args

    def tokenize_corpus(self, corpus):
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        return tokenized_corpus

    def split_doc(self, filename, split_size=100):
        # self.data[filename] -> Parsed OCR object
        # self.data.texts[filename] --> single string
        complete_texts = self.data.texts[filename]
       
        # format: [[s1, e1], [s2, e2], .... [sn, en]]
        # for each page, store the index of start and ending token in complete_texts
        token_indices_per_page = self.data.token_indices_per_page[filename]

        logging.info('Total pages in file: {}'.format(len(token_indices_per_page)))
        # simple splitting by newline
        # corpus = complete_texts.split("\n")
    
        # spaCy sentence splitter
        doc = NLP(complete_texts)
        corpus = [(sent.text.strip()) for sent in doc.sents]

        logging.info("Total Corpus Size {} docs".format(len(corpus)))
        doc_to_id_map = {}
        for id, doc in enumerate(corpus):
            doc_to_id_map[doc] = id

        # para splitter
        # OCR cluster (white spacing) -- see  IbocrTextProcessing.cluster_based_on_DIST in framework.py
        return corpus, doc_to_id_map

class Retrieval(ModelTrainer):
    def train(self, corpus, doc_to_id_map, tokenized_corpus):
        self.corpus = corpus  # list of documents
        self.model = BM25Okapi(tokenized_corpus)
        self.doc_to_id_map = doc_to_id_map

    def predict(self, query, len_results):
        tokenized_query = query.split(" ")
        doc_scores = self.model.get_scores(tokenized_query)
        # array([0.        , 0.93729472, 0.        ])
        logging.info("Scores available for {} docs".format(len(doc_scores)))

        top_docs = self.model.get_top_n(tokenized_query, self.corpus, n=len_results)
        top_scores = []
        for doc in top_docs:
            top_scores.append(doc_scores[self.doc_to_id_map[doc]])
        
        return top_docs, top_scores

    def analyze_result(self, results):
        pass