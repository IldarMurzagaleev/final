from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import numpy as np
import subprocess
from sklearn.preprocessing import StandardScaler
import tensorflow as tf



app = Flask(__name__)

cmd = 'python model.py'
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

df = pd.read_csv('ebw_data.csv')
X = df[['IW', 'IF', 'VW', 'FP']].to_numpy()
y = df[['Depth', 'Width']].to_numpy()
sc = StandardScaler()
X_train = sc.fit_transform(X)


@app.route('/', methods= ["GET", "POST"])
def hello_world():
    if request.method == 'POST':
        iwp = float(request.values.get('iw'))
        ifp = float(request.values.get('if'))
        vwp = float(request.values.get('vw'))
        fpp = float(request.values.get('fp'))

        model = tf.keras.models.load_model('saved_model/my_model')
        depth, width = model.predict(sc.transform(np.array([[iwp, ifp, vwp, fpp]])))


    return "Depth: {depth}, Width: {width}"

if __name__ == '__main__':
    app.run(debug=True)
