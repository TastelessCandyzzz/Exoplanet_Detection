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

# Prediction Route with Validation
@app.route("/predict", methods=["POST"])
def predict():
    # Get the single string of comma-separated values
    features_str = request.form.get("features")

    # Try to convert to numbers
    try:
        # Split by comma and convert each item to a float
        features = [float(x.strip()) for x in features_str.split(",")]

        # Make the prediction
        prediction_result = prediction.make_prediction(features)
        return render_template("index.html", prediction_text=f"Predicted class: {prediction_result}")

    except ValueError:
        # If conversion fails, return an error message
        error_msg = "Invalid input. Please enter only comma-separated numeric values."
        return render_template("index.html", error_text=error_msg)
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"An error occurred: {e}"
        return render_template("index.html", error_text=error_msg)


# Submission Page
@app.route("/submission")
def submission_page():
    return render_template("submission.html")

# Submit Data Route with Validation
@app.route("/submit_data", methods=["POST"])
def submit_data():
    try:
        # Try to convert all form values to floats
        values = [float(x) for x in request.form.values()]

        # Proceed if successful
        submission.add_data(values)
        retrain.train_model()
        return redirect(url_for("submission_page"))

    except ValueError:
        # If conversion fails, reload the page with an error message
        error_msg = "Invalid input. Please ensure all fields contain only numeric values."
        return render_template("submission.html", error_text=error_msg)


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return "No file uploaded", 400
    file = request.files["file"]
    if file.filename == "":
        return "Empty file name", 400

    filepath = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(filepath)

    submission.add_csv(filepath)
    retrain.train_model()
    return redirect(url_for("submission_page"))

# About Page
@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
