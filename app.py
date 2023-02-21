from flask import Flask, render_template, request, url_for, redirect, session
from py import FuzzyKNN
import pickle
import pandas as pd

app = Flask(__name__)
app = Flask(__name__, template_folder='temp')

model = pickle.load(open("FKNNModel.pkl", 'rb'))


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
    data = pd.DataFrame.from_dict(data)
    print(data.head())
    return "s"


if __name__ == '__main__':
    app.run()
