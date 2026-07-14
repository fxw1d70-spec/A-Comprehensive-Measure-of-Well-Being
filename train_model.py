"""
Train the HDI prediction model.

Pipeline: load -> impute missing values with column means -> label encode ->
split -> fit Linear Regression -> evaluate -> serialise with pickle.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

# Anchor paths to this file, not the shell's working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

RAW_FEATURES = [
    "Life_Expectancy",
    "Expected_Years_Schooling",
    "Mean_Years_Schooling",
    "GNI_per_Capita",
]

# The model is fitted on log(GNI) rather than raw GNI. UNDP's own HDI formula
# uses ln(GNI) because income has diminishing returns on human development:
# the first $1,000 matters enormously, the 60,000th barely moves the needle.
# A raw-dollar term forces one straight line through both regimes and badly
# underfits low-income countries.
FEATURES = [
    "Life_Expectancy",
    "Expected_Years_Schooling",
    "Mean_Years_Schooling",
    "GNI_log",
]
TARGET = "HDI"

RANDOM_STATE = 42


def engineer_features(df):
    """Add the log-income term the HDI definition is built on."""
    df["GNI_log"] = np.log(df["GNI_per_Capita"])
    return df


def preprocess(df):
    """Impute nulls with the column mean and label-encode the country column."""
    print("Missing values before imputation:")
    print(df[RAW_FEATURES + [TARGET]].isnull().sum().to_string())

    for col in RAW_FEATURES:
        if df[col].isnull().any():
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            print(f"  filled {col} nulls with mean = {mean_value:.2f}")

    # Drop any row still missing the target -- we cannot learn from those.
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    # Label encoding. Country is encoded so the mapping can be reused elsewhere
    # (e.g. the app's country lookup), but it is deliberately NOT a model
    # feature: integer country IDs carry no ordinal meaning and would make the
    # regression fit noise.
    encoder = LabelEncoder()
    df["Country_Code"] = encoder.fit_transform(df["Country"])

    print(f"\nRows after preprocessing: {len(df)}")
    return df, encoder


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df, encoder = preprocess(df)
    df = engineer_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train set: {X_train.shape[0]} rows | Test set: {X_test.shape[0]} rows")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    train_r2 = model.score(X_train, y_train)

    # Shuffle before folding. The CSV is sorted by HDI descending, so unshuffled
    # folds would each be a contiguous block of near-identical HDI values --
    # within-fold variance collapses and R2 goes sharply negative for reasons
    # that have nothing to do with model quality.
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

    print("\n" + "=" * 46)
    print("MODEL PERFORMANCE")
    print("=" * 46)
    print(f"R-Squared (train)   : {train_r2:.4f}")
    print(f"R-Squared (test)    : {r2:.4f}")
    print(f"Accuracy            : {r2 * 100:.2f}%")
    print(f"Mean Absolute Error : {mae:.4f}")
    print(f"RMSE                : {rmse:.4f}")
    print(f"5-fold CV R-Squared : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    print("\nRegression coefficients:")
    for name, coef in zip(FEATURES, model.coef_):
        print(f"  {name:<28} {coef:+.6f}")
    print(f"  {'Intercept':<28} {model.intercept_:+.6f}")

    with open(os.path.join(MODEL_DIR, "hdi_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    metrics = {
        "r2_test": round(r2, 4),
        "r2_train": round(train_r2, 4),
        "accuracy_pct": round(r2 * 100, 2),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "cv_r2": round(float(cv_scores.mean()), 4),
        "n_countries": int(len(df)),
        "coefficients": {n: round(float(c), 6) for n, c in zip(FEATURES, model.coef_)},
        "intercept": round(float(model.intercept_), 6),
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model, encoder and metrics to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
