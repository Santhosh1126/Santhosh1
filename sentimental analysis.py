pip install nltk scikit-learn
import nltk
import random
import string
from nltk.corpus import movie_reviews
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download NLTK data
nltk.download('movie_reviews')

# Load movie review data
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]
random.shuffle(docs)

# Convert list of words to raw text
texts = [" ".join(words) for words, label in docs]
labels = [1 if label == "pos" else 0 for words, label in docs]  # 1=pos, 0=neg

# Preprocess text: Lowercase, remove punctuation
def clean_text(text):
    text = text.lower()
    return text.translate(str.maketrans('', '', string.punctuation))

texts_clean = [clean_text(t) for t in texts]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42)

# Convert text to numerical features
vectorizer = CountVectorizer(stop_words='english', max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
clf = LogisticRegression()
clf.fit(X_train_vec, y_train)

# Evaluate model
predictions = clf.predict(X_test_vec)
print("Classification Report:")
print(classification_report(y_test, predictions))

# Predict sentiment of new input
def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = clf.predict(vec)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞= ")
    if user_input.lower() == "exit":
        break
    print("Sentiment:", predict_sentiment(user_input))
   python sentiment_analysis.py 