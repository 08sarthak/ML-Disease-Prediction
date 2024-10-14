import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load dataset
df = pd.read_csv('Training.csv')
logging.info('CSV file loaded successfully.')

logging.info('First few records of the dataset:\n%s', df.head())

# Features and target
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Label encoding for the target variable
logging.info('Starting label encoding...')
le = LabelEncoder()
y = le.fit_transform(y)
logging.info('Label encoding completed.')

# Splitting data into training and test sets (no stratification)
logging.info('Splitting data into training and test sets...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info('Data splitting completed.')

# Train the Multinomial Naive Bayes model
logging.info('Training the Multinomial Naive Bayes model...')
clf = MultinomialNB()
clf.fit(X_train, y_train)
logging.info('Model training completed.')

# Make predictions on the test set
logging.info('Making predictions on the test set...')
y_pred = clf.predict(X_test)
logging.info('Predictions completed.')

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
logging.info('Model accuracy: %.2f%%', accuracy * 100)

# Restrict the classification report to only the classes present in y_test
labels_in_test = sorted(set(y_test))  # Get the unique classes present in y_test

# Classification report
report = classification_report(y_test, y_pred, labels=labels_in_test, target_names=le.inverse_transform(labels_in_test), zero_division=0)
logging.info('Classification report:\n%s', report)

print('Classification Report:\n', report)

from sklearn.model_selection import LeaveOneOut

# Leave-One-Out Cross-Validation
logging.info('Performing Leave-One-Out Cross-Validation...')
loo = LeaveOneOut()
cross_val_scores = cross_val_score(clf, X, y, cv=loo, scoring='accuracy')
logging.info('Cross-validation scores: %s', cross_val_scores)
logging.info('Average cross-validation accuracy: %.2f%%', cross_val_scores.mean() * 100)

# Saving the trained model and label encoder
logging.info('Saving the trained model and label encoder...')

joblib.dump(clf, 'disease_prediction_model.joblib')
joblib.dump(le, 'label_encoder.joblib')
logging.info('All objects saved successfully.')
