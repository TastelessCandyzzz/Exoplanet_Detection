import os
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd

import prediction
import submission
import retrain

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features_str = request.form.get("features")
    try:
        features = [float(x.strip()) for x in features_str.split(",")]
        prediction_result = prediction.make_prediction(features)
        return render_template("index.html", prediction_text=f"Predicted class: {prediction_result}")
    except ValueError:
        error_msg = "Invalid input. Please enter only comma-separated numeric values."
        return render_template("index.html", error_text=error_msg)
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        return render_template("index.html", error_text=error_msg)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return render_template("index.html", error_text="No file part in the request.")
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error_text="No file selected.")

    if file and file.filename.endswith('.csv'):
        try:
            # --- NEW: Header Detection Logic ---
            # Read the first row to check for headers
            first_row = pd.read_csv(file, nrows=1, header=None).iloc[0]
            file.seek(0) # Reset file pointer to the beginning

            # Helper function to check if a value is numeric
            def is_numeric(val):
                try:
                    float(val)
                    return True
                except (ValueError, TypeError):
                    return False

            # If all values in the first row are non-numeric, assume it's a header
            is_header = all(not is_numeric(val) for val in first_row)

            # Read the full CSV based on whether a header was detected
            if is_header:
                df = pd.read_csv(file, header=0)
            else:
                df = pd.read_csv(file, header=None)
            # --- End of Header Detection Logic ---

            predictions = prediction.make_bulk_prediction(df)
            results_str = "\n".join(map(str, predictions))
            return render_template("index.html", bulk_prediction_results=results_str)
        except Exception as e:
            error_msg = f"Error processing CSV file: {e}"
            return render_template("index.html", error_text=error_msg)
    else:
        return render_template("index.html", error_text="Invalid file type. Please upload a CSV.")


@app.route("/submission")
def submission():
    return render_template("submission.html")

@app.route("/submit_data", methods=["POST"])
def submit_data():
    values_str = request.form.get("values")
    try:
        values = [float(x.strip()) for x in values_str.split(",")]
        submission.add_data(values)
        retrain.train_model()
        return redirect(url_for("submission"))
    except ValueError:
        error_msg = "Invalid input. Please enter comma-separated numeric values."
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
    return redirect(url_for("submission"))

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
