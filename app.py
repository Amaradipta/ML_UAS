import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("labeling.csv")

# Split data
X = data['contents']
y = data['sentiment_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train Naive Bayes model
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_counts, y_train)
y_pred_nb = nb_classifier.predict(X_test_counts)
accuracy_nb = accuracy_score(y_test, y_pred_nb)

# Display results
st.title("Sentiment Analysis with Naive Bayes")
st.write(f"Naive Bayes Accuracy: {accuracy_nb * 100:.2f}%")

cm_nb = confusion_matrix(y_test, y_pred_nb)
st.write("Confusion Matrix:")
st.write(cm_nb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='YlGnBu', xticklabels=['negative', 'neutral', 'positive'], yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
st.pyplot(plt.gcf())  # Use Streamlit to display matplotlib figure

# Classification report
target_names = ["negative", "neutral", "positive"]
report = classification_report(y_test, y_pred_nb, target_names=target_names, zero_division=1)
st.write("Classification Report:")
st.text(report)

# Function to plot classification report
def plot_classification_report(report, title='Classification report', cmap='RdYlGn'):
    lines = report.split('\n')
    classes = []
    plot_mat = []
    for line in lines[2: (len(lines) - 4)]:
        t = line.split()
        if len(t) > 0 and t[0] not in ('avg', 'accuracy', 'macro', 'weighted'):
            classes.append(t[0])
            v = [float(x) for x in t[1:-1]]
            plot_mat.append(v)
    plt.imshow(plot_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = ['precision', 'recall', 'f1-score']
    y_tick_marks = classes
    plt.xticks(range(len(x_tick_marks)), x_tick_marks)
    plt.yticks(range(len(classes)), classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Metrics')

plt.figure(figsize=(10, 6))
plot_classification_report(report)
st.pyplot(plt.gcf())  # Use Streamlit to display matplotlib figure

# Text input for user to predict sentiment
user_input = st.text_area("Enter text to analyze sentiment", "")
if user_input:
    user_input_counts = vectorizer.transform([user_input])
    user_pred = nb_classifier.predict(user_input_counts)
    st.write(f"Predicted Sentiment: {user_pred[0]}")