import pandas as pd
neurons = pd.read_csv('./data/raw/neurons.csv.gz')
conns = pd.read_csv('./data/raw/connections_princeton.csv.gz')  # Added .gz extension

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

print("Primary Cell Type — non-null count:", neurons["Primary Cell Type"].notna().sum())
print("Primary Cell Type — unique values:", neurons["Primary Cell Type"].nunique())

print("\nTop 30 cell types:")
print(neurons["Primary Cell Type"].value_counts().head(30))

print("\nSample T4/T5 rows:")
t4t5 = neurons[neurons["Primary Cell Type"].astype(str).str.match(r"^T4|^T5", na=False)]
print(f"T4/T5 count: {len(t4t5)}")
print(t4t5["Primary Cell Type"].value_counts())