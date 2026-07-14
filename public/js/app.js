// Client-side HDI prediction. Mirrors app.py exactly: same coefficients, same
// validation rules, same tier thresholds, same [0,1] clamp.

import { MODEL, COUNTRIES } from "./model.js";

const BOUNDS = {
  life_expectancy: [20, 100],
  expected_schooling: [0, 25],
  mean_schooling: [0, 20],
  gni: [100, 200000],
};

const LABELS = {
  life_expectancy: "Life expectancy",
  expected_schooling: "Expected years of schooling",
  mean_schooling: "Mean years of schooling",
  gni: "GNI per capita",
};

const TIERS = [
  [0.8, "Very High", "very-high",
    "Among the most developed nations, with strong outcomes across health, education and income simultaneously."],
  [0.7, "High", "high",
    "Solid development with room to grow. Typically one dimension lags the other two."],
  [0.55, "Medium", "medium",
    "An emerging economy. Targeted investment in healthcare, schooling or income generation could move the needle substantially."],
  [0, "Low", "low",
    "Significant development challenges across multiple dimensions. A priority candidate for policy intervention and development aid."],
];

export function predict({ life_expectancy, expected_schooling, mean_schooling, gni }) {
  const c = MODEL.coefficients;
  const raw =
    MODEL.intercept +
    c.life_expectancy * life_expectancy +
    c.expected_schooling * expected_schooling +
    c.mean_schooling * mean_schooling +
    c.gni_log * Math.log(gni);

  // Linear regression is unbounded; HDI is defined on [0, 1].
  return Math.min(1, Math.max(0, raw));
}

export function classify(score) {
  for (const [threshold, label, cssClass, description] of TIERS) {
    if (score >= threshold) return { label, cssClass, description };
  }
}

export function validate(values) {
  const errors = [];
  for (const [field, [low, high]] of Object.entries(BOUNDS)) {
    const v = values[field];
    if (v === "" || v === null || v === undefined) {
      errors.push(`${LABELS[field]} is required.`);
    } else if (!Number.isFinite(v)) {
      errors.push(`${LABELS[field]} must be a number.`);
    } else if (v < low || v > high) {
      errors.push(`${LABELS[field]} must be between ${low} and ${high}.`);
    }
  }
  if (
    errors.length === 0 &&
    values.mean_schooling > values.expected_schooling
  ) {
    errors.push("Mean years of schooling cannot exceed expected years of schooling.");
  }
  return errors;
}

// Each indicator's share of the total positive contribution, so the user can
// see which dimension carries the score and where the gap sits.
export function drivers(values) {
  const c = MODEL.coefficients;
  const raw = {
    "Life Expectancy": c.life_expectancy * values.life_expectancy,
    "Expected Schooling": c.expected_schooling * values.expected_schooling,
    "Mean Schooling": c.mean_schooling * values.mean_schooling,
    "Income (log GNI)": c.gni_log * Math.log(values.gni),
  };
  const total = Object.values(raw).reduce((a, b) => a + b, 0);
  if (total <= 0) return [];
  return Object.entries(raw)
    .map(([name, value]) => ({ name, pct: Math.round((1000 * value) / total) / 10 }))
    .sort((a, b) => b.pct - a.pct);
}

export { COUNTRIES, MODEL };
