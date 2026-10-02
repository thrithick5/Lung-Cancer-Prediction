import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

# Example probabilities (simulate predictions)
np.random.seed(42)

y_true = np.random.randint(0,2,1000)
y_prob = np.random.uniform(0,1,1000)

prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

plt.figure(figsize=(6,6))

plt.plot(prob_pred, prob_true, marker='o', label="Model Calibration")
plt.plot([0,1],[0,1], linestyle='--', label="Perfect Calibration")

plt.xlabel("Mean Predicted Probability")
plt.ylabel("True Probability")
plt.title("Calibration Reliability Diagram")

plt.legend()

plt.tight_layout()

plt.savefig("calibration_reliability_plot.png", dpi=300)

plt.show()

print("Calibration plot saved as calibration_reliability_plot.png")