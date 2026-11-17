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