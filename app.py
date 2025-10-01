import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np

import prediction
import submission
import retrain

app = Flask(__name__)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Prediction Route
@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction_result = prediction.make_prediction(features)
    return render_template("index.html", prediction_text=f"Predicted class: {prediction_result}")

# Submission Page
@app.route("/submission")
def submission_page():
    return render_template("submission.html")

@app.route("/submit_data", methods=["POST"])
def submit_data():
    values = [float(x) for x in request.form.values()]
    submission.add_data(values)  # Append to numpy dataset
    retrain.train_model()        # Retrain model
    return redirect(url_for("submission_page"))

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
