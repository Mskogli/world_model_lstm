# %% 
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use("seaborn-v0_8-talk")
plt.figure(figsize=(10, 5.5))

file_paths = [
    "1024-1-200-gaussians.npy",
    "1024-2-200-gaussians.npy",
    "1024-5-200-gaussians.npy",
    "1024-10-gaussians.npy",
    "1024-20-200-gaussians.npy"
]

labels = [
    r"$K=1$",
    r"$K=2$",
    r"$K=5$",
    r"$K=10$",
    r"$K=20$",
]

for file_path, label in zip(file_paths, labels):
    try:
        array = np.load(file_path)[0:200]
        x_values = np.arange(len(array))
        plt.plot(x_values, array, label=label)

    except FileNotFoundError:
        print(f"File not found: {file_path}")

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.legend()
plt.show()