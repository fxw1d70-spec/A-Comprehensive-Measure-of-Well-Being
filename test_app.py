"""End-to-end checks: the three scenarios from the brief, plus validation."""

import re

from app import app


def score_from(html):
    m = re.search(r'class="result-score">([\d.]+)<', html)
    return float(m.group(1)) if m else None


def tier_from(html):
    m = re.search(r'class="result-tier">([\w\s]+?) Human Development<', html)
    return m.group(1) if m else None


def main():
    client = app.test_client()
    failures = []

    # Pages load.
    for route in ["/", "/predict", "/insights"]:
        r = client.get(route)
        status = "OK" if r.status_code == 200 else f"FAIL ({r.status_code})"
        if r.status_code != 200:
            failures.append(f"{route} returned {r.status_code}")
        print(f"GET {route:<12} {status}")

    print("\n--- Scenarios from the brief ---")
    scenarios = [
        ("1. Very High (Norway-like)",
         {"life_expectancy": 83.2, "expected_schooling": 18.2,
          "mean_schooling": 13.0, "gni": 64660}, "Very High"),
        ("2. Medium (emerging economy)",
         {"life_expectancy": 67.2, "expected_schooling": 11.9,
          "mean_schooling": 6.7, "gni": 6590}, "Medium"),
        ("3. Low (intervention needed)",
         {"life_expectancy": 55.0, "expected_schooling": 7.0,
          "mean_schooling": 2.5, "gni": 1100}, "Low"),
    ]

    for name, payload, expected in scenarios:
        r = client.post("/predict", data=payload)
        score, tier = score_from(r.get_data(as_text=True)), tier_from(r.get_data(as_text=True))
        ok = tier == expected
        if not ok:
            failures.append(f"{name}: expected {expected}, got {tier}")
        print(f"{'PASS' if ok else 'FAIL'}  {name:<32} HDI {score}  -> {tier}")

    print("\n--- Accuracy against real reported HDI ---")
    checks = [
        ("Switzerland", 84.0, 16.5, 13.9, 66933, 0.962),
        ("Japan", 84.8, 15.2, 13.4, 42274, 0.925),
        ("Brazil", 72.8, 15.6, 8.1, 14370, 0.754),
        ("India", 67.2, 11.9, 6.7, 6590, 0.633),
        ("Niger", 61.9, 6.5, 2.1, 1240, 0.400),
    ]
    for country, le, es, ms, gni, actual in checks:
        r = client.post("/predict", data={
            "country": country, "life_expectancy": le, "expected_schooling": es,
            "mean_schooling": ms, "gni": gni,
        })
        pred = score_from(r.get_data(as_text=True))
        err = abs(pred - actual)
        flag = "OK" if err < 0.05 else "off"
        print(f"  {country:<13} predicted {pred:.3f} | actual {actual:.3f} | error {err:.3f}  {flag}")

    print("\n--- Validation (bad input must be rejected) ---")
    bad = [
        ("empty field", {"life_expectancy": "", "expected_schooling": 12,
                         "mean_schooling": 8, "gni": 5000}),
        ("non-numeric", {"life_expectancy": "abc", "expected_schooling": 12,
                         "mean_schooling": 8, "gni": 5000}),
        ("out of range", {"life_expectancy": 500, "expected_schooling": 12,
                          "mean_schooling": 8, "gni": 5000}),
        ("mean > expected", {"life_expectancy": 70, "expected_schooling": 6,
                             "mean_schooling": 15, "gni": 5000}),
        ("negative GNI", {"life_expectancy": 70, "expected_schooling": 12,
                          "mean_schooling": 8, "gni": -500}),
    ]
    for name, payload in bad:
        r = client.post("/predict", data=payload)
        ok = r.status_code == 400
        if not ok:
            failures.append(f"validation '{name}' was not rejected")
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} -> HTTP {r.status_code}")

    print("\n--- Autofill API ---")
    r = client.get("/api/country/Japan")
    print(f"  /api/country/Japan   -> {r.status_code} {r.get_json()}")
    r = client.get("/api/country/Atlantis")
    ok404 = r.status_code == 404
    if not ok404:
        failures.append("unknown country did not 404")
    print(f"  /api/country/Atlantis-> {r.status_code} (expected 404)")

    print("\n" + "=" * 50)
    if failures:
        print(f"{len(failures)} FAILURE(S):")
        for f in failures:
            print("  -", f)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
