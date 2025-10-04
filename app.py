import os
from flask import Flask, request, render_template, redirect, url_for, flash, session, Response
import numpy as np
import pandas as pd

import prediction
import submission
import retrain

app = Flask(__name__)
app.secret_key = 'your_super_secret_key' # Needed for flash messages

# --- NEW: Define the specific feature lists ---

# 17 features for the PREDICTION page
PREDICTION_FEATURES_INFO = [
    {"name": "koi_impact", "desc": "Impact parameter of the transit."},
    {"name": "koi_duration", "desc": "Duration of the transit in hours."},
    {"name": "koi_depth", "desc": "Depth of the transit in parts per million."},
    {"name": "koi_prad", "desc": "Planetary radius in Earth radii."},
    {"name": "koi_sma", "desc": "Orbit semi-major axis."},
    {"name": "koi_dor", "desc": "Distance over stellar radius."},
    {"name": "koi_max_sngle_ev", "desc": "Maximum single event significance."},
    {"name": "koi_max_mult_ev", "desc": "Maximum multiple event significance."},
    {"name": "koi_model_snr", "desc": "Transit signal-to-noise ratio."},
    {"name": "koi_count", "desc": "Number of observed transits."},
    {"name": "koi_steff", "desc": "Stellar effective temperature in Kelvin."},
    {"name": "koi_smet", "desc": "Stellar metallicity."},
    {"name": "koi_kepmag", "desc": "Kepler magnitude of the star."},
    {"name": "koi_fwm_srao", "desc": "FWM RA offset."},
    {"name": "koi_fwm_sdeco", "desc": "FWM Dec offset."},
    {"name": "koi_dicco_msky", "desc": "Difference image centroid offset from KIC (sky)."},
    {"name": "koi_dikco_msky", "desc": "Difference image KIC centroid offset (sky)."},
]

# 18 values for the SUBMISSION page (target + 17 features)
SUBMISSION_FEATURES_INFO = [
    {"name": "koi_disposition", "desc": "The target classification (e.g., 1 for CANDIDATE, 0 for FALSE POSITIVE)."}
] + PREDICTION_FEATURES_INFO


@app.route("/")
def index():
    # Pass the 17 prediction features to the index page
    session.pop('prediction_data', None)
    return render_template("index.html", features=PREDICTION_FEATURES_INFO)

@app.route("/submission")
def submission():
    # Pass all 18 submission values to the submission page
    return render_template("submission.html", features=SUBMISSION_FEATURES_INFO)


# --- (The rest of your app.py routes remain the same) ---

@app.route("/predict", methods=["POST"])
def predict():
    features_str = request.form.get("features")
    try:
        features = [float(x.strip()) for x in features_str.split(",")]
        # Add a check for the correct number of features
        if len(features) != 17:
            error_msg = f"Invalid input. Please enter exactly 17 comma-separated values. You entered {len(features)}."
            return render_template("index.html", error_text=error_msg, features=PREDICTION_FEATURES_INFO)

        prediction_result = prediction.make_prediction(features)
        return render_template("index.html", prediction_text=f"Predicted class: {prediction_result}", features=PREDICTION_FEATURES_INFO)
    except ValueError:
        error_msg = "Invalid input. Please enter only comma-separated numeric values."
        return render_template("index.html", error_text=error_msg, features=PREDICTION_FEATURES_INFO)
    except Exception as e:
        error_msg = f"An error occurred: {e}"
        return render_template("index.html", error_text=error_msg, features=PREDICTION_FEATURES_INFO)

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    if "file" not in request.files:
        return render_template("index.html", error_text="No file part in the request.", features=PREDICTION_FEATURES_INFO)
    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", error_text="No file selected.", features=PREDICTION_FEATURES_INFO)

    if file and file.filename.endswith('.csv'):
        try:
            # Header Detection Logic
            first_row = pd.read_csv(file, nrows=1, header=None).iloc[0]
            file.seek(0)
            def is_numeric(val):
                try: float(val); return True
                except (ValueError, TypeError): return False
            is_header = all(not is_numeric(val) for val in first_row)
            df = pd.read_csv(file, header=0 if is_header else None)

            if df.shape[1] != 17:
                error_msg = f"Invalid CSV. Please ensure it has exactly 17 feature columns. Your file has {df.shape[1]}."
                return render_template("index.html", error_text=error_msg, features=PREDICTION_FEATURES_INFO)

            # Make predictions
            predictions = prediction.make_bulk_prediction(df)

            # --- NEW: Prepare data for download ---
            # 1. Add predictions as a new column to the original data
            df['prediction'] = predictions

            # 2. Store the complete DataFrame (as a CSV string) in the user's session
            session['prediction_data'] = df.to_csv(index=False)

            # Format results for display in the textarea (only the predictions)
            results_str = "\n".join(map(str, predictions))

            # Pass a flag to the template to show the download button
            return render_template("index.html", bulk_prediction_results=results_str, features=PREDICTION_FEATURES_INFO, download_ready=True)

        except Exception as e:
            error_msg = f"Error processing CSV file: {e}"
            return render_template("index.html", error_text=error_msg, features=PREDICTION_FEATURES_INFO)
    else:
        return render_template("index.html", error_text="Invalid file type. Please upload a CSV.", features=PREDICTION_FEATURES_INFO)

@app.route('/download_predictions')
def download_predictions():
    # Get the CSV data from the session, clearing it after access
    csv_data = session.pop('prediction_data', None)

    if csv_data:
        # Create a response object with the CSV data
        response = Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition":
                     "attachment; filename=predictions.csv"}
        )
        return response
    else:
        # If no data is in the session, redirect to the home page
        return redirect(url_for('index'))


@app.route("/submit_data", methods=["POST"])
def submit_data():
    values_str = request.form.get("values")
    try:
        values = [float(x.strip()) for x in values_str.split(",")]
        # Add a check for the correct number of values
        if len(values) != 18:
            error_msg = f"Invalid input. Please enter exactly 18 comma-separated values. You entered {len(values)}."
            return render_template("submission.html", error_text=error_msg, features=SUBMISSION_FEATURES_INFO)

        submission.add_data(values)
        retrain.train_model()
        flash('Data submitted successfully! The model is now retraining.', 'success')
        return redirect(url_for("submission"))
    except ValueError:
        error_msg = "Invalid input. Please enter comma-separated numeric values."
        return render_template("submission.html", error_text=error_msg, features=SUBMISSION_FEATURES_INFO)

@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    # (This function remains the same, but you should ensure uploaded CSVs have 18 columns)
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
    flash('CSV uploaded successfully! The model is now retraining.', 'success')
    return redirect(url_for("submission"))

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
