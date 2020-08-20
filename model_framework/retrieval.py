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

        #sentenceIndices is an array in format [[s1, e1], [s2, e2], .... [sn, en]] which 
        #gives starting and ending indices for each sentence in corpus. 
        sentenceIndices = []
        startPointer = 0 #tracking pointer to fill out sentenceIndices
        endPointer = 0 #tracking pointer to fill out sentenceIndices

        for x in range(0, len(corpus)):
          sentence = corpus[x]
          endPointer = startPointer + len(sentence) - 1
          sentenceIndices.append([startPointer, endPointer])
          startPointer = endPointer + 2

        #corresponding_page_numbers is an array in format [p1, p2, p3, ..., pn] which 
        #gives the page number that each sentence in corpus is located on
        corresponding_page_numbers = []

        for x in range(0, len(corpus)):
          what_page = 0
          for page in token_indices_per_page:
            if ((sentenceIndices[x][0] >= page[0]) and (sentenceIndices[x][1] <= page[1])):
              what_page = token_indices_per_page.index(page) + 1
          if (what_page == 0): #this means it is split between two pages 
            for page in token_indices_per_page:
              if ((sentenceIndices[x][0] >= page[0]) and (sentenceIndices[x][0] <= page[1]) and (sentenceIndices[x][1] >= page[1])):
                what_page = token_indices_per_page.index(page) + 1
                break
          corresponding_page_numbers.append(what_page)
        
        logging.info("Total Corpus Size {} docs".format(len(corpus)))
        doc_to_id_map = {}

        for id, doc in enumerate(corpus):
            doc_to_id_map[doc] = id
        doc_to_pageNumber_map = {}

        for id, doc in enumerate(corpus):
          doc_to_pageNumber_map[doc] = corresponding_page_numbers[id]

        return corpus, doc_to_id_map, doc_to_pageNumber_map

class Retrieval(ModelTrainer):
    def train(self, corpus, tokenized_corpus, doc_to_id_map, doc_to_pageNumber_map):
        self.corpus = corpus  # list of documents
        self.model = BM25Okapi(tokenized_corpus)
        self.doc_to_id_map = doc_to_id_map
        self.doc_to_pageNumber_map = doc_to_pageNumber_map

    def predict(self, query, len_results):
        tokenized_query = query.split(" ")
        doc_scores = self.model.get_scores(tokenized_query)
        # array([0.        , 0.93729472, 0.        ])
        logging.info("Scores available for {} docs".format(len(doc_scores)))

        top_docs = self.model.get_top_n(tokenized_query, self.corpus, n=len_results)
        top_scores = []
        top_pageNumbers = []
        for doc in top_docs:
            top_scores.append(doc_scores[self.doc_to_id_map[doc]])
            top_pageNumbers.append(self.doc_to_pageNumber_map[doc]) 
        
        return top_docs, top_scores, top_pageNumbers

    def analyze_result(self, results):
        pass