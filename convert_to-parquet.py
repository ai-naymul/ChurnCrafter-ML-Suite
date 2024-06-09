import pandas as pd

df = pd.read_csv('data\Customer-Churn.csv')
df.to_parquet('data.parquet')