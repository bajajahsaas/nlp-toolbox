## Model Training Framework

- [ ] [PRD](https://instabase.atlassian.net/wiki/spaces/MI/pages/457277553/Model+Training+Framework+PRD)
- [ ] [JIRA Epic](https://instabase.atlassian.net/browse/INSIGHTS-1427)
- [ ] [Slides](https://docs.google.com/presentation/d/10mXA7K5sa_nAkqx2onsIfrH3TPj2Ni4LfCOxDhN5XBI/edit?usp=sharing)

### To Run

- Download dependencies as in [requirements.txt](requirements.txt)
- Change sdk and local build paths in framework.py
- Change Access tokens at start of each .ipynb file


### Use-Cases Supported

- Evaluate Refiner Results ([run_refiner.ipynb](run_refiner.ipynb))
- Infer BERT for name extraction ([run_NER.ipynb](run_NER.ipynb))
- Finetune BERT for name extraction ([train_NER.ipynb](train_NER.ipynb))
- Company Name extraction using Neural Nets ([run_cmp_name.ipynb](run_cmp_name.ipynb))
- Question Answering using BERT ([run_QA.ipynb](run_QA.ipynb))
- Question Answerring Demo ([run_QA_demo.ipynb](run_QA_demo.ipynb))
- Document Retrieval ([run_retrieval_demo.ipynb](run_retrieval_demo.ipynb))