from pathlib import Path

import matplotlib.pyplot as plt


def plot_outcome_counts(data):
    counts = data["Outcome"].value_counts().reindex([0, 1], fill_value=0)
    print(counts.to_string())

    output = Path("docs/figures/outcome_counts.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Outcome = 0", "Outcome = 1"], counts.to_numpy())
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, counts.max() * 1.12)
    ax.set_ylabel("Number of records")
    ax.set_title("Distribution of Diabetes Outcomes")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

def plot_missing_measurements(data):
    columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    measurements = data[columns]
    missing = (measurements.isna() | measurements.eq(0)).sum()
    missing = missing.sort_values(ascending=True)
    print(missing.sort_values(ascending=False).to_string())

    output = Path("docs/figures/missing_measurements.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(missing.index, missing.to_numpy(), color="#287A78")
    labels = [f"{count} ({count / len(data):.1%})" for count in missing]
    ax.bar_label(bars, labels=labels, padding=4)
    ax.set_xlim(0, missing.max() * 1.30)
    ax.set_xlabel("Number of records with a missing measurement")
    ax.set_title("Missing Measurements by Variable")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig) 