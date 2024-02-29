# `emotion-classifier`
A simple classification model that tries to answer the following question: What emotion (anger, fear, joy, love, sadness, surprise) does a message (sentence) display?

## Motivation (Why)
1. As an ML platform engineer, [Jay](https://www.linkedin.com/in/shilongjaycui/) wants to be able to empathize with his customers (data scientists) by becoming a data scientist himself.
2. Jay wants to answer data science questions related to mental health.

## Workflow (What)
#### Step 1: Get data.
- Hugging Face `dair-ai/emotion` dataset, which can be found [here](https://huggingface.co/datasets/dair-ai/emotion)

#### Step 2: Featurize the data.
- scikit-learn's [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) vectorizer (`sklearn.feature_extraction.text.TfidfVectorizer`), which can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer)

#### Step 3: Train a model.
- sciklit-learn's random forest classifier (`sklearn.ensemble.RandomForestClassifier`), which can be found [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier)

#### Step 4: Evaluate the model.
- accuracy score
- classification report
- precision-recall curves
- confusion matrix

#### Next steps:
- Further evaluate the model by creating a [data map](https://nbertagnolli.medium.com/838c235cd702).
- Figure out if the current model is too smart of too dumb by plotting learning curves.
- Augment the data (especially the "surprise" category, the most difficult one to classify) with more samples and more features, possibly using ChatGPT.
- Gain insights into why the model is having a hard time telling the difference bewteen joy (1) and love (2).

## Installation (How)
1. Clone this repo on your local machine:
   ```bash
   $ git clone git@github.com:shilongjaycui/emotion-classifier.git
   ```
2. Navigate into the cloned repo:
   ```bash
   $ cd emotion-classifier
   ```
3. Create a virtual environment and activate it:
   ```bash
   $ python -m venv venv
   $ source venv/bin/activate
   ```
4. Install the dependencies, which are listed in Makefile:
   ```bash
   $ make install
   ```
5. Navigate into the Python package (`src/emotion_classifier`):
   ```bash
   $ cd src/emotion_classifier
   ```
6. Do exploratory data analysis ([EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis)):
   ```bash
   $ python analyze_data.py
   ```
7. Train the model:
   ```bash
   $ python train_model.py
   ```
8. Evaluate the model:
   ```bash
   $ python evaluate_model.py
   ```
