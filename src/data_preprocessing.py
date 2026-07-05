import numpy as np

data = np.load("data/student_task_adl_prepared_dataset.npz")

X = data["X"]
y = data["y"]

# Add dimension to match requested input shape
X = X[..., np.newaxis]

print(X.shape, y.shape)

# compute random baseline to beat

baseline = y.mean()
print(f"Random baseline (to beat): {baseline}")