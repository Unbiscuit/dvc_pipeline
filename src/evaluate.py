import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os, sys, json


df = pd.read_csv('df_after_preprocessing.csv')
y = df['Survived']
df = df.loc[:, df.columns != 'Survived']
df = (df-df.mean())/df.std()
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
model = tf.keras.models.load_model('model.keras')

results = model.evaluate(X_test, y_test)

with open('scores.json', 'w') as f:
    json.dump({'acc': results[1]}, f)
