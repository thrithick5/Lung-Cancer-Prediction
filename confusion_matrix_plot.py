import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example confusion matrix values (replace with your real values if available)

cm = np.array([
    [182, 8],
    [11, 199]
])

labels = ["Benign", "Malignant"]

plt.figure(figsize=(6,5))

sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Lung Cancer Classification")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)

plt.show()

print("Confusion matrix saved as confusion_matrix.png")