from flask import Flask, render_template, request, url_for, redirect,session
from py import FuzzyKNN

app = Flask(__name__)
app = Flask(__name__,template_folder='temp')

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

if __name__=='__main__':
  app.run()