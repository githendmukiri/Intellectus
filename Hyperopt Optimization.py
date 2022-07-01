import re

import pandas as pd


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb

import warnings

warnings.filterwarnings("ignore")

# load data
print('Loading training data...')
train = pd.read_csv('train.csv')
print('Finished.')

# preprocessing
print('preprocessing...')
stop_words = stopwords.words('english')
wnl = WordNetLemmatizer()


def preprocess(text_column):
    new_comment = []
    for comment in text_column:
        text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(comment).lower()).strip()
        text = [wnl.lemmatize(i) for i in text.split(' ') if i not in stop_words]
        new_comment.append(' '.join(text))
        return new_comment

    train['comment_text'] = preprocess(train['comment_text'])


print('Finished.')
# Vectorization
print('Vectorization...')

X = train.drop(columns=['toxic'])
print(X)
y = train.loc[:, 'toxic']

cv = CountVectorizer(binary=True)
cv.fit(train['comment_text'])

X = cv.transform(X['comment_text'])

print('Finished.')

# split into test and train
print('Split Data...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print('Finished.')


def main():
    space = {
        'max_depth': hp.quniform('max_depth', 3, 28, 1),
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'n_estimators': hp.uniform('n_estimators', 100, 10000),
        'gamma': hp.quniform('gamma', 0, 1, 0.025),
        'min_child_weight': hp.quniform('min_child_weight', 1, 30, .5),
        'subsample': hp.quniform('subsample', 0.0025, 1, 0.025),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1.0, 0.025)
    }

    trials = Trials()

    best = fmin(fn=hyperparameter_tuning,
                space=space,
                algo=tpe.suggest,
                max_evals=50,
                trials=trials)

    print(best)


def hyperparameter_tuning(space):
    print('Building Model...')
    model = xgb.XGBClassifier(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        gamma=space['gamma'],
        min_child_weight=space['min_child_weight'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        random_state=42,
        tree_method='gpu_hist',
    )

    evaluation = [(X_train, y_train), (X_test, y_test)]

    model.fit(X_train, y_train,
              eval_set=evaluation, eval_metric="rmse",
              early_stopping_rounds=10, verbose=False)
    print('Finished.')

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred > 0.5)
    print("SCORE:", accuracy)
    # change the metric if you like
    return {'loss': -accuracy, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':
    main()
