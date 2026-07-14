# Human Development Index (HDI) Predictor

A Flask web application that predicts a country's Human Development Index from four
indicators — life expectancy, expected years of schooling, mean years of schooling, and
GNI per capita — and classifies it into one of the UNDP's four development tiers.

Built with Python, scikit-learn, Flask, pandas, matplotlib and seaborn.

## Quick start

```bash
pip install -r requirements.txt

python eda.py          # exploratory analysis -> static/images/
python train_model.py  # train + evaluate + pickle -> model/
python app.py          # serve at http://127.0.0.1:5000
```

The trained model is committed, so you can skip straight to `python app.py` if you just
want to run the app. Run `python test_app.py` to verify everything end to end.

## Model performance

| Metric | Value |
|---|---|
| R² (test set) | 0.9557 |
| R² (training set) | 0.9957 |
| 5-fold cross-validated R² | 0.9853 ± 0.015 |
| Mean absolute error | 0.0143 |
| RMSE | 0.0335 |

Spot-checked against real reported HDI values, predictions land within 0.013:

| Country | Predicted | Actual |
|---|---|---|
| Switzerland | 0.964 | 0.962 |
| Japan | 0.925 | 0.925 |
| Brazil | 0.748 | 0.754 |
| India | 0.632 | 0.633 |
| Niger | 0.413 | 0.400 |

## Two decisions worth knowing about

**Income enters the model as log(GNI), not raw GNI.** This is the single biggest driver
of the model's accuracy, and it comes straight from how UNDP defines HDI. Income has
sharply diminishing returns on human development — the first $1,000 per capita transforms
lives, the 60,000th barely registers. Fitting a straight line to raw dollars forces one
slope through both regimes and badly underfits low-income countries. Adding the log term
moved test R² from **0.887 to 0.956** and halved mean absolute error.

**Cross-validation folds must be shuffled.** `cross_val_score(model, X, y, cv=5)` reported
an R² of **-0.36** on this dataset. That was not a bad model — `data/hdi_data.csv` is
sorted by HDI descending, so unshuffled folds each become a contiguous block of
near-identical HDI values. Within-fold variance collapses, and R² (which is measured
*relative to the variance of the fold*) goes sharply negative. Using
`KFold(shuffle=True)` gives a stable 0.985 ± 0.015. This is an easy and very common trap
with any sorted dataset.

## Project structure

```
data/hdi_data.csv      190 countries, UNDP Human Development Report 2021–22
eda.py                 EDA: strip plots, distributions, heatmap, scatter, pairplot
train_model.py         Preprocessing -> Linear Regression -> pickle
app.py                 Flask app (routes, validation, tier classification)
model/                 Serialised model, label encoder, metrics.json
templates/             Home, prediction form, result, insights
static/                CSS + generated EDA figures
test_app.py            End-to-end checks
```

## Pipeline

1. **Preprocessing** — nulls filled with column means (the dataset has genuine gaps:
   Somalia and Eritrea lack schooling data, Monaco and Liechtenstein lack life
   expectancy). Rows missing the *target* are dropped rather than imputed.
2. **Label encoding** — `Country` is label-encoded and the encoder is pickled, but it is
   deliberately **not** a model feature. Integer country IDs carry no ordinal meaning
   (Albania=1, Algeria=2 does not imply Algeria is "twice" Albania), so feeding them to a
   regression just invites it to fit noise.
3. **Feature engineering** — `GNI_log = ln(GNI_per_Capita)`.
4. **Split** — 80/20 train/test, `random_state=42`.
5. **Train** — `LinearRegression`, evaluated with R², MAE, RMSE and shuffled 5-fold CV.
6. **Serialise** — pickled to `model/hdi_model.pkl`.

## Development tiers

| Tier | HDI range |
|---|---|
| Very High | ≥ 0.800 |
| High | 0.700 – 0.799 |
| Medium | 0.550 – 0.699 |
| Low | < 0.550 |

## App features

- **Prediction form** with server-side validation — rejects non-numeric input,
  out-of-range values, and the physically impossible case of mean years of schooling
  exceeding expected years of schooling.
- **Country autofill** — pick a country to populate the form with its real reported
  indicators and see how close the model lands.
- **Score breakdown** — each indicator's share of the predicted score, so you can see
  which dimension carries it and where the development gap sits.
- **Insights page** — the generated EDA figures with the model's learned coefficients.
- Predictions are clamped to [0, 1]; linear regression is unbounded but HDI is not.

## Notes on the data

`data/hdi_data.csv` was compiled from the **published UNDP Human Development Report
2021–22** figures rather than downloaded from Kaggle (which requires an account and API
token). The countries and indicator values are real. If you want to swap in a Kaggle
dataset later, keep the same column names and every script will work unchanged.

Because HDI is itself computed from these four indicators, high R² is expected here —
the model is recovering a known relationship, not discovering a hidden one. That makes it
a good pedagogical dataset but a poor test of predictive difficulty.
