import pickle

with open('preprocessing/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


print("Scaler expects input shape:", scaler.mean_.shape)

print("Means:\n", scaler.mean_)
print("Variances:\n", scaler.var_)

print(type(scaler))

import numpy as np

sample_input = np.array([[0.0, 50.0, 0.0, 0.0, 5000.0, 0.0] + [200.0]*19])  # dummy input
scaled_input = scaler.transform(sample_input)
print("Scaled input:\n", scaled_input)
