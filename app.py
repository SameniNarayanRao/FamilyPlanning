from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("knn.pkl", "rb"))

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])
def home():
    AG = request.form['AG']
    HLAP = request.form['HLAP']
    COF = request.form['COF']
    array = np.array([[AG, HLAP, COF]])      
    pred = model.predict(array)
    return render_template("result.html", df = pred)

if __name__ == "__main__":
    app.run(debug = True)