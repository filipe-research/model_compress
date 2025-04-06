import pandas as pd

df = pd.read_csv("dataset/annotations.txt", header=None)
print(f"Shape: {df.shape}")
print(df.head(3))