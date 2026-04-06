import pandas as pd

df1 = pd.read_csv('vindr_detection1.csv')
df2 = pd.read_csv('vindr_detection2.csv')
df = pd.concat([df1, df2], ignore_index=True)
df.to_csv('vindr_detection.csv', index=False)
print(len(df))