"""
Exploratory Data Analysis for the HDI dataset.

Generates strip plots, distribution plots, a correlation heatmap and scatter
plots into static/images/ so they can be surfaced in the Flask dashboard.
"""

import os

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed; we only write files
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Anchor paths to this file, not the shell's working directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_data.csv")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")

FEATURES = [
    "Life_Expectancy",
    "Expected_Years_Schooling",
    "Mean_Years_Schooling",
    "GNI_per_Capita",
]

sns.set_theme(style="whitegrid")


def hdi_tier(score):
    """UNDP's four development tiers."""
    if score >= 0.800:
        return "Very High"
    if score >= 0.700:
        return "High"
    if score >= 0.550:
        return "Medium"
    return "Low"


def load_data():
    df = pd.read_csv(DATA_PATH)
    print("Shape:", df.shape)
    print("\nFirst rows:\n", df.head())
    print("\nInfo:")
    df.info()
    print("\nSummary statistics:\n", df.describe())
    print("\nMissing values per column:\n", df.isnull().sum())
    return df


def plot_distributions(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, FEATURES):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
        ax.set_title(f"Distribution of {col.replace('_', ' ')}")
    fig.suptitle("Distribution of HDI Indicators", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "distributions.png"), dpi=110)
    plt.close(fig)


def plot_heatmap(df):
    corr = df[FEATURES + ["HDI"]].corr()
    print("\nCorrelation matrix:\n", corr.round(3))

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Matrix of HDI Indicators", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "heatmap.png"), dpi=110)
    plt.close(fig)


def plot_strip(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Low", "Medium", "High", "Very High"]
    sns.stripplot(
        data=df,
        x="Tier",
        y="Life_Expectancy",
        order=order,
        hue="Tier",
        hue_order=order,
        palette="viridis",
        size=7,
        jitter=0.25,
        legend=False,
        ax=ax,
    )
    ax.set_title("Life Expectancy by Development Tier", fontweight="bold")
    ax.set_xlabel("HDI Tier")
    ax.set_ylabel("Life Expectancy (years)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "stripplot.png"), dpi=110)
    plt.close(fig)


def plot_scatter(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, FEATURES):
        sns.scatterplot(
            data=df, x=col, y="HDI", hue="Tier",
            hue_order=["Low", "Medium", "High", "Very High"],
            palette="viridis", ax=ax, s=45, edgecolor="white", linewidth=0.4,
        )
        sns.regplot(
            data=df, x=col, y="HDI", scatter=False,
            color="#C44E52", line_kws={"linewidth": 1.5}, ax=ax,
        )
        ax.set_title(f"HDI vs {col.replace('_', ' ')}")
        ax.legend(fontsize=7, title=None)
    fig.suptitle("HDI vs Each Indicator", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "scatterplots.png"), dpi=110)
    plt.close(fig)


def plot_pairplot(df):
    g = sns.pairplot(
        df[FEATURES + ["HDI", "Tier"]].dropna(),
        hue="Tier",
        hue_order=["Low", "Medium", "High", "Very High"],
        palette="viridis",
        corner=True,
        plot_kws={"s": 28, "alpha": 0.8},
    )
    g.figure.suptitle("Pairwise Relationships", y=1.01, fontweight="bold")
    g.savefig(os.path.join(IMG_DIR, "pairplot.png"), dpi=100)
    plt.close(g.figure)


def main():
    os.makedirs(IMG_DIR, exist_ok=True)
    df = load_data()
    df["Tier"] = df["HDI"].apply(hdi_tier)

    print("\nCountries per tier:\n", df["Tier"].value_counts())

    plot_distributions(df)
    plot_heatmap(df)
    plot_strip(df)
    plot_scatter(df)
    plot_pairplot(df)

    print(f"\nSaved 5 figures to {IMG_DIR}/")


if __name__ == "__main__":
    main()
