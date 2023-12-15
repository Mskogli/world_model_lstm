import matplotlib.pyplot as plt

import numpy as np

# Replace 'your_file.npy' with the path to your .npy file
file_path = '1024-100ep.npy'

# Load the array from the .npy file
array = np.load(file_path)
print(array)

# Create the x-axis values (0, 1, 2, ..., N)
x_values = np.arange(len(array))

# Plotting the array
plt.plot(x_values, array)
plt.xlabel('Index')
plt.ylabel('Array Value')
plt.title('Plot of Array from .npy File')
plt.show()