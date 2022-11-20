# Simple Linear Regression

'''
This model predicts thet thickness and depth of the weld.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import tensorflow as tf

def build_and_compile_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(2)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

if __name__ == "__main__":
    # Importing the dataset
    df = pd.read_csv('ebw_data.csv')
    X = df[['IW', 'IF', 'VW', 'FP']].to_numpy()
    y = df[['Depth', 'Width']].to_numpy()
    sc = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    regressor = build_and_compile_model()
    history = regressor.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        verbose=1, epochs=100)

    result = regressor.evaluate(X_test, y_test, verbose=1)

    # Saving model using pickle
    regressor.save('saved_model')

    # Loading model to compare the results
    model = tf.keras.models.load_model('saved_model')
    pred = model.predict(sc.transform(np.array([[47, 139, 4.5, 80]])))
    print(pred[0][0], pred[0][1])
    print(result)
