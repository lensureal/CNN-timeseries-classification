import numpy as np
import matplotlib.pyplot as plt

data = np.load("data/student_task_adl_prepared_dataset.npz")

print(data.files)
X = data["X"]
y = data["y"]

print(X.shape, y.shape) # Dimensionen
print(np.unique(y, return_counts=True)) # Anteile stable/stall


plt.figure(figsize=(10, 4))
plt.plot(X[np.where(y==0)][3], label='Stable') # Erstes Stable-Beispiel
plt.plot(X[np.where(y==1)][7], label='Stall')  # Erstes Stall-Beispiel
plt.legend()
plt.show()