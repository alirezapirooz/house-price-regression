import pandas as pd



df = pd.read_csv("houseprice.csv")

print("Initial shape:", df.shape)



df["Area"] = pd.to_numeric(df["Area"], errors="coerce")
df = df.dropna()
df = df[(df["Area"] > 60) & (df["Area"] < 400)]

print("After area cleaning:", df.shape)

df.to_csv("cleaned_data.csv", index=False)

print("Saved to cleaned_data.csv")