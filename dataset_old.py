# dataset.py

import pandas as pd
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib


from utils import preprocess_text


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


try:
    df = pd.read_csv('synthetic_disease_dataset.csv')
    logging.info('CSV file loaded successfully.')
except FileNotFoundError:
    logging.error('CSV file not found. Please check the file path.')
    raise

logging.info('First few records of the dataset:\n%s', df.head())


logging.info('Starting text preprocessing...')
df['processed_description'] = df['description'].apply(preprocess_text)
logging.info('Text preprocessing completed.')


logging.info('Starting TF-IDF vectorization...')
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_description'])
logging.info('TF-IDF vectorization completed.')


logging.info('Starting label encoding...')
le = LabelEncoder()
y = le.fit_transform(df['disease'])
logging.info('Label encoding completed.')


logging.info('Splitting data into training and test sets with stratification...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
logging.info('Data splitting completed.')


logging.info('Training the Multinomial Naive Bayes model...')
clf = MultinomialNB()
clf.fit(X_train, y_train)
logging.info('Model training completed.')


logging.info('Making predictions on the test set...')
y_pred = clf.predict(X_test)
logging.info('Predictions completed.')


accuracy = accuracy_score(y_test, y_pred)
logging.info('Model accuracy: %.2f%%', accuracy * 100)

report = classification_report(
    y_test, y_pred, target_names=le.classes_, zero_division=0
)
logging.info('Classification report:\n%s', report)

print('Classification Report:\n', report)


logging.info('Performing Stratified K-Fold Cross-Validation...')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
logging.info('Cross-validation scores: %s', cross_val_scores)
logging.info('Average cross-validation accuracy: %.2f%%', cross_val_scores.mean() * 100)


logging.info('Saving the trained model and preprocessing objects...')
joblib.dump(clf, 'disease_prediction_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(le, 'label_encoder.joblib')
logging.info('All objects saved successfully.')
