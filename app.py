from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("../model.pkl", "rb"))

fertilizers = [
"Urea",
"DAP",
"14-35-14",
"28-28",
"17-17-17",
"20-20",
"10-26-26"
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])
    moisture = float(request.form["Moisture"])
    soil = int(request.form["Soil"])
    crop = int(request.form["Crop"])
    nitrogen = float(request.form["Nitrogen"])
    phosphorous = float(request.form["Phosphorous"])
    potassium = float(request.form["Potassium"])

    features = np.array([[temp,humidity,moisture,soil,crop,nitrogen,phosphorous,potassium]])

    prediction = model.predict(features)

    result = fertilizers[prediction[0]]

    return render_template("index.html", prediction_text="Recommended Fertilizer: "+result)

if __name__ == "__main__":
    app.run(debug=True)