import pandas as pd
import numpy as np

df = pd.read_csv('raw_data/train.csv')

df = df.loc[:, df.columns != 'Name']
df = df.loc[:, df.columns != 'Embarked']
df = df.loc[:, df.columns != 'PassengerId']
df = df.loc[:, df.columns != 'Ticket']

df['Sex'] = pd.factorize(df['Sex'])[0]

letter_to_index = {'G': 1, 'F': 2, 'E': 3, 'D': 4, 'C': 5, 'B': 6, 'A': 7}
df['Cabin'] = df['Cabin'].map(lambda x: letter_to_index.get(x[0], np.nan) if pd.notna(x) else np.nan)

def use_lr_to_a_column(df, column):
    from sklearn.linear_model import LinearRegression
    test_data = df[df[column].isnull()]
    df.dropna(inplace=True)
    x_train = df.drop(column,axis=1)
    x_train = x_train.drop('Survived',axis=1)
    y_train = df[column]
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    x_test = test_data[['Pclass','Sex', 'SibSp', 'Parch', 'Fare']]
    y_pred = lr.predict(x_test)
    return y_pred

def fill_nan_with_lr(df):
    df_without_age = df.loc[:, df.columns != 'Age']
    df_without_cabin = df.loc[:, df.columns != 'Cabin']
    a = pd.Series(use_lr_to_a_column(df_without_age, 'Cabin'))
    b = pd.Series(use_lr_to_a_column(df_without_cabin, 'Age'))

    nan_indices = df[df['Cabin'].isna()].index
    for index, replacement_value in zip(nan_indices, a):
        df.at[index, 'Cabin'] = replacement_value

    nan_indices = df[df['Age'].isna()].index
    for index, replacement_value in zip(nan_indices, b):
        df.at[index, 'Age'] = replacement_value

    return df

df = fill_nan_with_lr(df)

df.to_csv("df_after_preprocessing.csv", index=False)  