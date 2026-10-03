import pandas as pd
neurons = pd.read_csv('neurons.csv.gz')
conns = pd.read_csv('connections_princeton.csv.gz')  # Added .gz extension

print("neurons shape:", neurons.shape)
print("connections shape:", conns.shape)


print("\nFirst 5 rows of neurons:")
print(neurons.head())

print("\nFirst 5 rows of connections:")
print(conns.head())

print("\nSample neurons:")
print(neurons.head(3))

print("\nSample connections:")
print(conns.head(3))