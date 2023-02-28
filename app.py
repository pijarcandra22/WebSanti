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


def change(x):
    if x == 0:
        return 1
    else:
        return 0


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
    hasil = pd.read_csv("dataHasil.csv")
    hasil['Risiko'] = hasil['Risiko'].apply(change)
    return render_template("patientdata.html", hasil=hasil)


@app.route('/riskresult')
def riskresult():
    return render_template("riskresult.html")


@app.route('/fknn', methods=['POST'])
def fknn():
    data = request.form.to_dict(flat=False)
    print(data['age'])

    data = pd.DataFrame.from_dict(data)
    backup = data.copy()
    dt = scaler.transform(data.values)
    data = pd.DataFrame(dt, columns=data.columns)

    hasil = model.predict(data)
    print(hasil)
    backup["Risiko"] = np.array(hasil[0])
    backup.columns = ['Umur', 'Tinggi' 'Badan', 'BB',
                      'LILA', 'sistolik', 'diastolik', 'Risiko']

    backup = pd.concat([backup, pd.read_csv("dataHasil.csv")], axis=0)
    backup.to_csv("dataHasil.csv", index=False)

    return str(hasil[0])


if __name__ == '__main__':
    app.run()
