# Model Training Framework

This repository aims to provide a unified structure to build custom ML models (powered by state-of-the-art NLP systems) to solve specific use-cases. The overall goal is to build tooling/a framework around loading data/goldens, training models, evaluating/analyzing those models, and exporting the models for usage.
- [ ] [PRD](https://instabase.atlassian.net/wiki/spaces/MI/pages/457277553/Model+Training+Framework+PRD)
- [ ] [JIRA Epic](https://instabase.atlassian.net/browse/INSIGHTS-1427)
- [ ] [Slides](https://docs.google.com/presentation/d/10mXA7K5sa_nAkqx2onsIfrH3TPj2Ni4LfCOxDhN5XBI/edit?usp=sharing)

### To Run

- Install dependencies as in [requirements.txt](requirements.txt)
- Change sdk and local build paths in [framework.py](framework.py)
- Change Access token at the start of each .ipynb file
- Follow markdown comments in the notebook files to download models and data

### Use-Cases Supported

- Evaluate Refiner Results ([run_refiner.ipynb](run_refiner.ipynb))
- Infer BERT for name extraction ([run_NER.ipynb](run_NER.ipynb))
- Finetune BERT for name extraction ([train_NER.ipynb](train_NER.ipynb))
- Company Name extraction using Neural Nets ([run_cmp_name.ipynb](run_cmp_name.ipynb))
- Question Answering using BERT ([run_QA.ipynb](run_QA.ipynb))
- Question Answerring Demo ([run_QA_demo.ipynb](run_QA_demo.ipynb))
- Document Retrieval ([run_retrieval_demo.ipynb](run_retrieval_demo.ipynb))

### Models

- Evaluation of Refiner Results ([refiner.py](refiner.py))
- BERT for sequence classification ([bert_ner.py](bert_ner.py))
- Question Answering using BERT ([bert_qa.py](bert_qa.py))
- Keras Neural Nets ([MultiLayerPerceptron.py](MultiLayerPerceptron.py))
- Document Retrieval ([retrieval.py](retrieval.py))

### Backbone framework code

- Framework ([framework.py](framework.py)
- BERT finetuning using huggingface ([bert_finetuning.py](bert_finetuning.py))
- BERT inference using huggingface ([infer_bert_classifier.py](infer_bert_classifier.py), [infer_bert_qa.py](infer_bert_qa.py))
- Utilities for BERT ([bert_utils.py](bert_utils.py))
- Text Preprocessing ([preprocessing.py](preprocessing.py), [rule_features.py](rule_features.py))
### Steps to add a new model or use-case

