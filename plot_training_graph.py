import matplotlib.pyplot as plt

# Example values (replace with your real training logs if available)

epochs = list(range(1, 21))

train_accuracy = [
0.72,0.78,0.82,0.85,0.88,0.90,0.91,0.92,0.93,0.94,
0.945,0.947,0.948,0.949,0.951,0.952,0.953,0.954,0.955,0.956
]

val_accuracy = [
0.70,0.76,0.80,0.83,0.86,0.88,0.89,0.90,0.91,0.92,
0.923,0.925,0.927,0.928,0.930,0.931,0.933,0.934,0.935,0.936
]

train_loss = [
0.82,0.70,0.60,0.52,0.45,0.40,0.36,0.32,0.29,0.26,
0.24,0.22,0.21,0.20,0.19,0.18,0.17,0.16,0.15,0.14
]

val_loss = [
0.85,0.73,0.63,0.56,0.50,0.45,0.41,0.37,0.34,0.31,
0.29,0.27,0.26,0.25,0.24,0.23,0.22,0.21,0.20,0.19
]


plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(epochs, train_accuracy, label='Training Accuracy')
plt.plot(epochs, val_accuracy, label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy During Training")
plt.legend()

plt.subplot(1,2,2)
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss During Training")
plt.legend()

plt.tight_layout()

plt.savefig("training_graph.png", dpi=300)

plt.show()