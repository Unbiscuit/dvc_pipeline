import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

df = pd.read_csv('df_after_preprocessing.csv')

y = df['Survived']
df = df.loc[:, df.columns != 'Survived']

df = (df-df.mean())/df.std()

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(32, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
    ),
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",
        monitor="val_loss",
        save_best_only=True,
)
]

params = yaml.safe_load(open('params.yaml'))['train']
batch_size = params['batch_size']

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

history = model.fit(X_train,
                    y_train,
                    epochs=20,
                    callbacks=callbacks_list,
                    batch_size=batch_size,
                    validation_split=0.2)

model.save('model.keras')
