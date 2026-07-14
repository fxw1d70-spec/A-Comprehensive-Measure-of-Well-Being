"""Flask web application for the HDI Predictor."""

import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Anchor paths to this file, not the shell's working directory, so the app runs
# from any folder (e.g. VS Code's Run button, which sets cwd to the workspace root).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_data.csv")

FEATURES = [
    "Life_Expectancy",
    "Expected_Years_Schooling",
    "Mean_Years_Schooling",
    "GNI_log",
]

# Valid input ranges, used for server-side validation. Loosely bracket the
# observed real-world ranges so users cannot push the model far outside the
# data it was fitted on and get a meaningless extrapolation back.
BOUNDS = {
    "life_expectancy": (20.0, 100.0),
    "expected_schooling": (0.0, 25.0),
    "mean_schooling": (0.0, 20.0),
    "gni": (100.0, 200000.0),
}

TIERS = [
    (0.800, "Very High", "very-high",
     "Among the most developed nations, with strong outcomes across health, "
     "education and income simultaneously."),
    (0.700, "High", "high",
     "Solid development with room to grow. Typically one dimension lags the "
     "other two."),
    (0.550, "Medium", "medium",
     "An emerging economy. Targeted investment in healthcare, schooling or "
     "income generation could move the needle substantially."),
    (0.000, "Low", "low",
     "Significant development challenges across multiple dimensions. A "
     "priority candidate for policy intervention and development aid."),
]


def load_model():
    with open(os.path.join(MODEL_DIR, "hdi_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    return model, metrics


model, metrics = load_model()
countries_df = pd.read_csv(DATA_PATH)


def classify(score):
    """Map an HDI score to its UNDP development tier."""
    for threshold, label, css_class, description in TIERS:
        if score >= threshold:
            return {"label": label, "css_class": css_class, "description": description}


def build_drivers(values):
    """Per-feature contribution to the prediction, for the result breakdown.

    Each indicator's contribution is coefficient * value; we report each as a
    share of the total positive contribution so the user can see which
    dimension is carrying (or dragging) the score.
    """
    coefs = dict(zip(FEATURES, model.coef_))
    raw = {
        "Life Expectancy": coefs["Life_Expectancy"] * values["life_expectancy"],
        "Expected Schooling": coefs["Expected_Years_Schooling"] * values["expected_schooling"],
        "Mean Schooling": coefs["Mean_Years_Schooling"] * values["mean_schooling"],
        "Income (log GNI)": coefs["GNI_log"] * np.log(values["gni"]),
    }
    total = sum(raw.values())
    if total <= 0:
        return []
    return sorted(
        [
            {"name": k, "value": round(v, 4), "pct": round(100 * v / total, 1)}
            for k, v in raw.items()
        ],
        key=lambda d: d["pct"],
        reverse=True,
    )


@app.route("/")
def home():
    return render_template("index.html", metrics=metrics)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    countries = sorted(countries_df["Country"].dropna().unique().tolist())

    if request.method == "GET":
        return render_template("predict.html", countries=countries, metrics=metrics)

    # Parse and validate every field before touching the model.
    values, errors = {}, []
    labels = {
        "life_expectancy": "Life expectancy",
        "expected_schooling": "Expected years of schooling",
        "mean_schooling": "Mean years of schooling",
        "gni": "GNI per capita",
    }

    for field, (low, high) in BOUNDS.items():
        raw = (request.form.get(field) or "").strip()
        if not raw:
            errors.append(f"{labels[field]} is required.")
            continue
        try:
            num = float(raw)
        except ValueError:
            errors.append(f"{labels[field]} must be a number.")
            continue
        if not np.isfinite(num) or not (low <= num <= high):
            errors.append(f"{labels[field]} must be between {low:g} and {high:g}.")
            continue
        values[field] = num

    # Mean schooling cannot exceed expected schooling -- that combination is
    # not physically meaningful and the model has never seen it.
    if not errors and values["mean_schooling"] > values["expected_schooling"]:
        errors.append(
            "Mean years of schooling cannot exceed expected years of schooling."
        )

    if errors:
        return render_template(
            "predict.html",
            countries=countries,
            metrics=metrics,
            errors=errors,
            form=request.form,
        ), 400

    features = pd.DataFrame(
        [[
            values["life_expectancy"],
            values["expected_schooling"],
            values["mean_schooling"],
            np.log(values["gni"]),
        ]],
        columns=FEATURES,
    )

    # Linear regression is unbounded; HDI is defined on [0, 1]. Clamp so an
    # extreme input can never surface an impossible score like 1.04.
    score = float(np.clip(model.predict(features)[0], 0.0, 1.0))

    return render_template(
        "result.html",
        score=round(score, 3),
        tier=classify(score),
        drivers=build_drivers(values),
        inputs=values,
        country=(request.form.get("country") or "").strip() or None,
        metrics=metrics,
    )


@app.route("/insights")
def insights():
    return render_template("insights.html", metrics=metrics)


@app.route("/api/country/<name>")
def country_data(name):
    """Autofill endpoint: return a country's real indicators from the dataset."""
    row = countries_df[countries_df["Country"].str.lower() == name.lower()]
    if row.empty:
        return {"error": "Country not found"}, 404

    record = row.iloc[0]

    def clean(v):
        return None if pd.isna(v) else float(v)

    return {
        "country": record["Country"],
        "life_expectancy": clean(record["Life_Expectancy"]),
        "expected_schooling": clean(record["Expected_Years_Schooling"]),
        "mean_schooling": clean(record["Mean_Years_Schooling"]),
        "gni": clean(record["GNI_per_Capita"]),
        "actual_hdi": clean(record["HDI"]),
    }


if __name__ == "__main__":
    app.run(debug=True, port=5000)
