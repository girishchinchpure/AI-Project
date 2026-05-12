# Sentiment Analysis Project (IMDB Reviews)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, classification_report


# Load dataset
df = pd.read_csv("IMDB Dataset.csv")

print("Dataset loaded successfully!\n")
print(df.head())


# Preprocessing

# Convert labels to numeric
df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.2, random_state=42
)


# Text Vectorization

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model

model = MultinomialNB()
model.fit(X_train_vec, y_train)


# Predictions

pred = model.predict(X_test_vec)


# Evaluation

print("\nModel Accuracy:", accuracy_score(y_test, pred))

print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# Test on custom input

def predict_review(text):
    text_vec = vectorizer.transform([text])
    result = model.predict(text_vec)

    if result[0] == 1:
        print("Positive Review 😊")
    else:
        print("Negative Review 😞")


# Example
print("\nCustom Test:")
predict_review("This movie was amazing and very entertaining!")