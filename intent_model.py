# intent_model.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# דוגמאות לאימון מודל זיהוי כוונה (Intent Classification)
training_data = [
    ("hi", "greeting"),
    ("hello", "greeting"),
    ("hey", "greeting"),
    ("goodbye", "goodbye"),
    ("bye", "goodbye"),
    ("thanks", "thanks"),
    ("thank you", "thanks"),
    ("show me more", "more_results"),
    ("I want to see more", "more_results"),
    ("change preferences", "change_prefs"),
    ("adjust my choices", "change_prefs"),
    ("start over", "restart"),
    ("I'd like to restart", "restart"),
    ("I'm done", "goodbye")
]

texts, labels = zip(*training_data)

# בניית pipeline עם TF-IDF ו־Logistic Regression
intent_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# אימון המודל
intent_classifier.fit(texts, labels)

