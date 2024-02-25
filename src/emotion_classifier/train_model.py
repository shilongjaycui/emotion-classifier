from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

from analyze_data import X, y

MODEL_FNAME: str = "model.joblib"

model: Pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', RandomForestClassifier())
])

if __name__ == '__main__':
    print("Model steps:")
    for name, step in model.named_steps.items():
        print(f"Step: {name}")
        print(step)
        print("\n")

    print("Training model...")
    model.fit(X, y)
    print("Model trained.")

    print("Saving model...")
    dump(value=model, filename=MODEL_FNAME)
    print("Model saved.")
