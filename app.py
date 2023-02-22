from flask import Flask, render_template, request, url_for, redirect, session
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from py import FuzzyKNN
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
app = Flask(__name__, template_folder='temp')

scaler = StandardScaler()
scaler.fit(pd.read_csv("DatasetTugasAkhir.csv").iloc[:, :-1].to_numpy())

model = pickle.load(open("model2.sav", 'rb'))


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/beranda')
def beranda():
    return render_template("beranda.html")


@app.route('/inputdata')
def inputdata():
    return render_template("inputdata.html")


@app.route('/patientdata')
def patientdata():
    return render_template("patientdata.html")


@app.route('/riskresult')
def riskresult():
    return render_template("riskresult.html")


@app.route('/fknn', methods=['POST'])
def fknn():
    data = request.form.to_dict(flat=False)
    print(data['age'])

    data = pd.DataFrame.from_dict(data)
    dt = scaler.transform(data.values)
    data = pd.DataFrame(dt, columns=data.columns)

    hasil = model.predict(data)
    print(hasil)
    return str(hasil[0])


if __name__ == '__main__':
    app.run()
