import os, sys, re, math, statistics as stat
from typing import TextIO

import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.svm import SVC, SVR

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def main(train_io: TextIO, test_io: TextIO, out_io: TextIO):
    # Read raw data
    train_df, test_df = read_input(train_io, test_io)

    # Extract features
    x, y, x_test = extract_features(train_df, test_df)

    # Define the model
    model = make_pipeline(
        StandardScaler(),
        SVC()
    )
    # model = make_pipeline(
    #     StandardScaler(),
    #     PolynomialFeatures(degree=2),
    #     LogisticRegression(max_iter=200, fit_intercept=False)
    # )

    # Learn
    model: Pipeline = model.fit(x, y)

    # Score
    score_model(model, x, y)

    # Use
    predictions = predict(model, x_test)

    # Write predictions
    write_output(out_io, predictions)


def col_names(base_str, start_ix, end_inclusive_ix):
    return [base_str + str(i) for i in range(start_ix, end_inclusive_ix + 1)]


def read_input(train_io: TextIO, test_io: TextIO) -> (DataFrame, DataFrame):
    # cat_cols = ['Sex', 'Deck'] + col_names('Feature', 1, 10)
    # id_cols = ['PassengerId', 'Name']
    # target = 'Survived'
    # type_map = {c: str for c in cat_cols + id_cols}

    train_df = pd.read_csv(train_io)   # dtype=type_map
    test_df = pd.read_csv(test_io)
    return train_df, test_df


def extract_features(train_df: DataFrame, test_df: DataFrame):
    train_xy_df = extract_features_df(train_df)
    test_x_df = extract_features_df(test_df)

    # features_correlation = train_xy_df.corr().abs()

    train_x_df: DataFrame = train_xy_df.drop(columns='Survived')
    train_y_df = train_xy_df['Survived']

    train_test_x_df = pd.concat([train_x_df, test_x_df])

    imputer = IterativeImputer(max_iter=100, n_nearest_features=2)
    imputer.fit(train_test_x_df)
    train_x = imputer.transform(train_x_df)
    test_x = imputer.transform(test_x_df)

    return train_x, train_y_df, test_x


def extract_features_df(df: DataFrame) -> DataFrame:
    # These features are not correlated with Y
    df: DataFrame = df.drop(columns=['PassengerId', 'Ticket', 'Name'])
    df['Sex'] = df['Sex'].apply(lambda v: 1 if v == 'male' else 0)
    df['Deck'] = df['Cabin'].apply(cabin_to_deck_cat)
    df = df.drop(columns=['Cabin'])
    df['Embarked'] = df['Embarked'].fillna(value='S')
    df['Embarked_Sex'] = df.apply(lambda row: f"{row['Embarked']}_{row['Sex']}", axis=1)
    df = df.drop(columns=['Embarked'])

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySizeBucket'] = df['FamilySize'].apply(lambda v: family_size_bucket(v))
    df = df.drop(columns=['Parch', 'SibSp', 'FamilySize'])

    familysize_hot_encoded_df = pd.get_dummies(df['FamilySizeBucket'], prefix='FamilySizeBucket')
    df = df.join(familysize_hot_encoded_df)
    df = df.drop(columns='FamilySizeBucket')

    # embarked_hot_encoded_df = pd.get_dummies(df['Embarked'], prefix='Embarked')
    # df = df.join(embarked_hot_encoded_df)
    # df = df.drop(columns='Embarked')

    deck_hot_encoded_df = pd.get_dummies(df['Deck'], prefix='Deck')
    df = df.join(deck_hot_encoded_df)
    df = df.drop(columns='Deck')

    embarked_sex_hot_encoded_df = pd.get_dummies(df['Embarked_Sex'], prefix='Embarked_Sex')
    df = df.join(embarked_sex_hot_encoded_df)
    df = df.drop(columns=['Embarked_Sex'])

    df = df.drop(columns=['Embarked_Sex_C_1'], errors='ignore')

    return df


def cabin_to_deck_cat(cabin) -> str:
    if pd.notna(cabin):
        deck_letter = re.sub(r'[^A-Z]', '', cabin)[0]
        return dict(A='ABC', B='ABC', C='ABC', D='DE', E='DE', F='FG', G='FG', T='ABC')[deck_letter]
    else:
        return 'M'


def family_size_bucket(size: int) -> str:
    if size == 1:
        return 'Alone'
    if size in (2, 3, 4):
        return 'Small'
    return 'Large'


def score_model(model, x, y):
    result = cross_val_score(model, x, y)
    print()
    print(f'  Mean CV score: {result.mean()}')
    print(f'Stdev CV scores: {result.std()}')
    print(f'  Var CV scores: {result.std()**2}')
    print(f'  All CV scores: {result}')
    print(f'Lowest CV score: {min(result)}')


def predict(model: Pipeline, x_challenges):
    return model.predict(x_challenges)


def write_output(file: TextIO, predictions: list[float]):
    file.write('PassengerId,Survived\n')
    for i, pred in enumerate(predictions):
        file.write(f'{i+892},{round(pred, 2)}\n')


if __name__ == '__main__':
    with open('input/train.csv') as train, \
         open('input/test.csv') as test,   \
         open('output.csv', 'w') as output:
        main(train, test, output)
