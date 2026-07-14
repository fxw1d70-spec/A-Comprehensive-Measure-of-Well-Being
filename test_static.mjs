// Verify the browser model reproduces the Python model exactly.
// Run: node test_static.mjs

import { predict, classify, validate, COUNTRIES } from "./public/js/app.js";

// Expected values produced by the Python/scikit-learn model (test_app.py).
const CASES = [
  ["Scenario 1: Very High", { life_expectancy: 83.2, expected_schooling: 18.2, mean_schooling: 13.0, gni: 64660 }, 0.961, "Very High"],
  ["Scenario 2: Medium", { life_expectancy: 67.2, expected_schooling: 11.9, mean_schooling: 6.7, gni: 6590 }, 0.632, "Medium"],
  ["Scenario 3: Low", { life_expectancy: 55.0, expected_schooling: 7.0, mean_schooling: 2.5, gni: 1100 }, 0.386, "Low"],
  ["Switzerland", { life_expectancy: 84.0, expected_schooling: 16.5, mean_schooling: 13.9, gni: 66933 }, 0.964, "Very High"],
  ["Japan", { life_expectancy: 84.8, expected_schooling: 15.2, mean_schooling: 13.4, gni: 42274 }, 0.925, "Very High"],
  ["Brazil", { life_expectancy: 72.8, expected_schooling: 15.6, mean_schooling: 8.1, gni: 14370 }, 0.748, "High"],
  ["India", { life_expectancy: 67.2, expected_schooling: 11.9, mean_schooling: 6.7, gni: 6590 }, 0.632, "Medium"],
  ["Niger", { life_expectancy: 61.9, expected_schooling: 6.5, mean_schooling: 2.1, gni: 1240 }, 0.413, "Low"],
];

let failures = 0;

console.log("--- JS predictions vs Python model ---");
for (const [name, input, expected, expectedTier] of CASES) {
  const score = Number(predict(input).toFixed(3));
  const tier = classify(score).label;
  const match = score === expected && tier === expectedTier;
  if (!match) failures++;
  console.log(
    `${match ? "PASS" : "FAIL"}  ${name.padEnd(22)} js ${score.toFixed(3)} | python ${expected.toFixed(3)} | ${tier}`
  );
}

console.log("\n--- Validation parity ---");
const BAD = [
  ["empty field", { life_expectancy: "", expected_schooling: 12, mean_schooling: 8, gni: 5000 }],
  ["non-numeric", { life_expectancy: NaN, expected_schooling: 12, mean_schooling: 8, gni: 5000 }],
  ["out of range", { life_expectancy: 500, expected_schooling: 12, mean_schooling: 8, gni: 5000 }],
  ["mean > expected", { life_expectancy: 70, expected_schooling: 6, mean_schooling: 15, gni: 5000 }],
  ["negative GNI", { life_expectancy: 70, expected_schooling: 12, mean_schooling: 8, gni: -500 }],
];
for (const [name, input] of BAD) {
  const errs = validate(input);
  const ok = errs.length > 0;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(16)} -> rejected: ${ok}`);
}

console.log("\n--- Clamping ---");
const huge = predict({ life_expectancy: 100, expected_schooling: 25, mean_schooling: 20, gni: 200000 });
const tiny = predict({ life_expectancy: 20, expected_schooling: 0, mean_schooling: 0, gni: 100 });
const clamped = huge <= 1 && tiny >= 0;
if (!clamped) failures++;
console.log(`${clamped ? "PASS" : "FAIL"}  extremes stay in [0,1]: max=${huge.toFixed(3)} min=${tiny.toFixed(3)}`);

console.log(`\n--- Data ---\nCountries loaded: ${COUNTRIES.length}`);

console.log("\n" + "=".repeat(50));
console.log(failures === 0 ? "All checks passed. JS matches Python exactly." : `${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
