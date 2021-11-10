import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score

from nltk.tokenize import RegexpTokenizer


def clean_str(string, reg=RegexpTokenizer(r'[a-z]+')):
    # Clean a string with RegexpTokenizer
    string = string.lower()
    tokens = reg.tokenize(string)
    return " ".join(tokens)


def logistic_regression_classification():
    ham_df = pd.read_csv("/home/rajpatel/Documents/NLP/email_classification/enron_csv/ham_emails.csv")
    ham_df = ham_df[ham_df['label'] == 1]

    spam_df = pd.read_csv("/home/rajpatel/Documents/NLP/email_classification/enron_csv/spam_emails.csv")
    spam_df = spam_df[spam_df['label'] == 0]

    df = ham_df.append(spam_df)

    # Create a new column with the cleaned messages
    df['clean_data'] = df['data'].apply(lambda string: clean_str(string))

    # Convert a collection of text documents to a matrix of token counts
    cv = CountVectorizer()

    X = cv.fit_transform(df.clean_data)
    y = df.label

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # LogisticRegression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)

    ascore = accuracy_score(y_test, y_pred)
    print("\n\n>> Accuracy Score: ", ascore*100)

    pscore = precision_score(y_test, y_pred, average='binary')
    print("\n\n>> Precision Score: ", pscore*100)

    rscore = recall_score(y_test, y_pred, average='binary')
    print("\n\n>> Recall Score: ", rscore*100)

    # ------------------------------------------- Sample Email Prediction ----------------------------------------------

    df = pd.read_csv("/home/rajpatel/Documents/NLP/email_classification/enron_csv/sample.csv")

    df['clean_data'] = df['data'].apply(lambda string: clean_str(string))

    msg = df['clean_data']

    data = cv.transform(msg)

    result = logreg.predict(data)

    print("\n\n>> Email Data: ", df['data'].iloc[0])
    print("\n\n>> Actual Label: ", df['label'].iloc[0])
    print("\n\n>> Predicted Label: ", result[0])


if __name__ == '__main__':
    logistic_regression_classification()
